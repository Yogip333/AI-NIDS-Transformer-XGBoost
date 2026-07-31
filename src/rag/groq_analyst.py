# Adapted from concepts by:
# Grattafiori, A. et al. (2024) 'The Llama 3 Herd of Models',
#   arXiv:2407.21783.

"""LLM-assisted alert enrichment via the Groq inference API.

Once the retrieval layer in ``threat_rag.py`` has pulled the closest
knowledge-base entries for a detection, this module sends the detection
context, the matched entries and the session statistics to a Groq-hosted
Llama 3.3 70B model (Grattafiori et al., 2024) and asks for a short
structured JSON assessment: false-positive likelihood, a contextual
threat summary, a handful of concrete investigation steps and a severity
note. The generator is strictly downstream of the ML pipeline and never
alters a prediction; any failure in the LLM path is caught and logged so
the alert still goes out with the retrieval-only context.
"""
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_TIMEOUT = 10   # seconds — fast inference, keep API latency low

SYSTEM_PROMPT = """\
You are a senior SOC (Security Operations Centre) analyst specialising in network intrusion detection.
You receive structured alerts from an ML-based NIDS and produce concise, actionable assessments.

ALWAYS respond with valid JSON and exactly these keys:
{
  "fp_likelihood":       "LOW" | "MEDIUM" | "HIGH",
  "fp_reasoning":        "<one sentence — why this may or may not be a false positive>",
  "threat_summary":      "<2-3 sentences contextualising the threat based on the SPECIFIC session data>",
  "investigation_steps": ["<step 1>", "<step 2>", "<step 3>"],
  "severity_context":    "<one sentence on actual risk given the confidence score and session stats>"
}

Rules:
- Do NOT repeat generic textbook descriptions — focus on the specific numbers in the session stats.
- investigation_steps must be concrete actions (e.g. "Check firewall logs for src IP at..."), not vague.
- fp_likelihood should be HIGH when confidence < 0.70 or stats don't match expected attack pattern.
- Keep every field brief and actionable. Respond with JSON only — no markdown, no extra text.\
"""

# Stat keys shown to the LLM and their human labels
_STAT_LABELS = {
    "stat_syn_ratio":        "SYN flag ratio",
    "stat_rst_ratio":        "RST flag ratio",
    "stat_port_entropy":     "Destination port entropy",
    "stat_unique_dst_ports": "Unique destination ports",
    "stat_avg_flow_bps":     "Avg flow bandwidth (bps)",
    "stat_short_flow_ratio": "Short flow ratio (<1s)",
    "stat_avg_duration":     "Avg flow duration (s)",
    "stat_avg_fwd_pkts":     "Avg forward packets/flow",
    "stat_pkt_len_mean":     "Avg packet length (bytes)",
    "stat_flow_count":       "Flows in session",
}


class GroqAnalyst:
    """
    Wraps the Groq chat-completions API to provide LLM-powered alert enrichment.

    Parameters
    ----------
    api_key : Groq API key (reads GROQ_API_KEY env var if not provided)
    model   : Groq model ID (default: llama-3.3-70b-versatile)
    timeout : HTTP timeout in seconds
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model:   str = GROQ_MODEL,
        timeout: int = GROQ_TIMEOUT,
    ):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._model   = model
        self._timeout = timeout
        self._client  = None

        if not self._api_key:
            logger.warning("GROQ_API_KEY not set — Groq enrichment disabled")
        else:
            self._init_client()

    def _init_client(self) -> None:
        try:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
            logger.info("Groq client initialised (model=%s)", self._model)
        except ImportError:
            logger.error("groq package not installed — run: pip install groq")
        except Exception as e:
            logger.error("Groq client init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        attack_name:     str,
        confidence:      float,
        session_stats:   dict,
        kb_context:      str,
        top_predictions: list[dict],
    ) -> dict:
        """
        Call Groq LLM and return enriched analysis dict.

        Falls back to an empty dict on any error so the rest of the alert
        pipeline is never blocked by a Groq failure.
        """
        if not self.available:
            return {}

        prompt = _build_user_prompt(
            attack_name, confidence, session_stats, kb_context, top_predictions
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,       # low temperature → consistent, factual output
                max_tokens=512,
                timeout=self._timeout,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_response(raw)
        except Exception as e:
            logger.warning("Groq analysis failed: %s", e)
            return {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_user_prompt(
    attack_name:     str,
    confidence:      float,
    session_stats:   dict,
    kb_context:      str,
    top_predictions: list[dict],
) -> str:
    """Format the user-turn prompt from detection data."""
    preds_text = ", ".join(
        f"{p['attack_name']} ({p['probability']:.1%})"
        for p in (top_predictions or [])[:3]
    )

    stat_lines = []
    for key, label in _STAT_LABELS.items():
        val = session_stats.get(key)
        if val is not None and float(val) != 0.0:
            stat_lines.append(f"  {label}: {float(val):.4f}")
    stats_block = "\n".join(stat_lines) if stat_lines else "  (no session statistics available)"

    # Trim KB context to keep prompt within token budget
    ctx_snippet = kb_context[:600] if kb_context else "(no KB context)"

    return (
        f"DETECTION RESULT:\n"
        f"  Attack type : {attack_name}\n"
        f"  Confidence  : {confidence:.1%}\n"
        f"  Top ML preds: {preds_text}\n\n"
        f"SESSION STATISTICS:\n{stats_block}\n\n"
        f"THREAT KNOWLEDGE BASE CONTEXT:\n{ctx_snippet}\n\n"
        f"Provide your SOC analyst assessment as JSON."
    )


def _parse_response(raw: str) -> dict:
    """Parse JSON from Groq response; return empty dict on failure."""
    # Strip markdown code fences if the model wraps in ```json ... ```
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    try:
        data = json.loads(text)
        # Validate expected keys exist
        required = {"fp_likelihood", "fp_reasoning", "threat_summary",
                    "investigation_steps", "severity_context"}
        if not required.issubset(data.keys()):
            logger.warning("Groq response missing keys: %s", required - data.keys())
        return data
    except json.JSONDecodeError as e:
        logger.warning("Could not parse Groq JSON: %s | raw=%s", e, raw[:200])
        return {}
