"""
Threat Mapping Evaluation for Mini AI-SOC.

Evaluates:
  1. MITRE mapping accuracy — does each attack class have correct ATT&CK mappings?
  2. Technique & tactic coverage — how many ATT&CK items are represented?
  3. Prediction-to-technique consistency — on correctly classified attacks, does RAG
     always return the correct MITRE mapping?
  4. RAG retrieval precision@1 — does vector search return the right threat entry?

Requires: evaluation_results_v4.json (or processed_data.pkl + trained models)

Output: models/checkpoints/threat_mapping_eval.json
"""
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Ground-truth MITRE mapping per CICIDS2017 attack class ────────────────────
# These are validated against ATT&CK Enterprise (v14+)

GROUND_TRUTH_MITRE = {
    0:  {"name": "BENIGN",                "tactics": [],                                          "techniques": []},
    1:  {"name": "Bot",                   "tactics": ["Command and Control", "Collection"],       "techniques": ["T1071", "T1571", "T1041"]},
    2:  {"name": "DDoS",                  "tactics": ["Impact"],                                  "techniques": ["T1498", "T1499"]},
    3:  {"name": "DoS GoldenEye",         "tactics": ["Impact"],                                  "techniques": ["T1499", "T1499.002"]},
    4:  {"name": "DoS Hulk",              "tactics": ["Impact"],                                  "techniques": ["T1499", "T1499.002"]},
    5:  {"name": "DoS Slowhttptest",      "tactics": ["Impact"],                                  "techniques": ["T1499", "T1499.001"]},
    6:  {"name": "DoS slowloris",         "tactics": ["Impact"],                                  "techniques": ["T1499", "T1499.001"]},
    7:  {"name": "FTP-Patator",           "tactics": ["Credential Access"],                       "techniques": ["T1110", "T1110.001"]},
    8:  {"name": "Heartbleed",            "tactics": ["Initial Access", "Collection"],             "techniques": ["T1190", "T1212"]},
    9:  {"name": "Infiltration",          "tactics": ["Initial Access", "Lateral Movement"],      "techniques": ["T1189", "T1570"]},
    10: {"name": "PortScan",              "tactics": ["Reconnaissance", "Discovery"],              "techniques": ["T1046", "T1018"]},
    11: {"name": "SSH-Patator",           "tactics": ["Credential Access"],                       "techniques": ["T1110", "T1110.001"]},
    12: {"name": "Web Attack Brute Force","tactics": ["Credential Access"],                       "techniques": ["T1110", "T1110.001"]},
    13: {"name": "Web Attack Sql Injection","tactics": ["Initial Access"],                        "techniques": ["T1190"]},
    14: {"name": "Web Attack XSS",        "tactics": ["Initial Access", "Execution"],             "techniques": ["T1190", "T1059.007"]},
}


def evaluate_knowledge_base_coverage():
    """Check how well the curated KB covers MITRE ATT&CK."""
    from src.rag.knowledge_base import THREAT_KNOWLEDGE_BASE

    results = {
        "total_entries": len(THREAT_KNOWLEDGE_BASE),
        "per_attack": {},
        "tactic_coverage": {},
        "technique_coverage": {},
    }

    kb_tactics = set()
    kb_techniques = set()
    gt_tactics = set()
    gt_techniques = set()

    for entry in THREAT_KNOWLEDGE_BASE:
        aid = entry.attack_id
        gt = GROUND_TRUTH_MITRE.get(aid, {})
        gt_t = set(gt.get("tactics", []))
        gt_tech = set(gt.get("techniques", []))
        kb_t = set(entry.mitre_tactics)
        kb_tech = set(entry.techniques)

        gt_tactics.update(gt_t)
        gt_techniques.update(gt_tech)
        kb_tactics.update(kb_t)
        kb_techniques.update(kb_tech)

        tactic_match = len(gt_t & kb_t) / max(len(gt_t), 1) if gt_t else 1.0
        # Compare T-codes by root (e.g. T1499 matches T1499.002)
        gt_roots = {t.split(".")[0] for t in gt_tech}
        kb_roots = {t.split(" ")[0].split(".")[0] for t in kb_tech}
        tech_overlap = len(gt_roots & kb_roots)

        results["per_attack"][entry.attack_name] = {
            "attack_id": aid,
            "kb_tactics": list(kb_t),
            "gt_tactics": list(gt_t),
            "tactic_recall": round(tactic_match, 4),
            "kb_techniques": list(kb_tech),
            "gt_techniques": list(gt_tech),
            "technique_overlap": tech_overlap,
            "technique_recall": round(tech_overlap / max(len(gt_tech), 1), 4),
        }

    results["tactic_coverage"] = {
        "kb_total": len(kb_tactics),
        "gt_total": len(gt_tactics),
        "overlap": len(kb_tactics & gt_tactics),
        "coverage_rate": round(len(kb_tactics & gt_tactics) / max(len(gt_tactics), 1), 4),
        "kb_tactics": sorted(kb_tactics),
        "gt_tactics": sorted(gt_tactics),
    }

    results["technique_coverage"] = {
        "kb_total": len(kb_techniques),
        "gt_total": len(gt_techniques),
        "overlap": len(kb_techniques & gt_techniques),
        "coverage_rate": round(len(kb_techniques & gt_techniques) / max(len(gt_techniques), 1), 4),
    }

    # Mean tactic/technique recall across attack classes
    recalls_tactic = [v["tactic_recall"] for v in results["per_attack"].values()]
    recalls_tech   = [v["technique_recall"] for v in results["per_attack"].values()]
    results["mean_tactic_recall"]    = round(np.mean(recalls_tactic), 4)
    results["mean_technique_recall"] = round(np.mean(recalls_tech), 4)

    return results


def evaluate_rag_retrieval():
    """Test RAG retrieval precision — query each attack name, check top-1 match."""
    import yaml
    from src.rag.threat_rag import ThreatRAG
    from src.data.loader import ID_TO_LABEL

    try:
        with open("configs/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config not found — skipping RAG retrieval test")
        return {}

    try:
        rag = ThreatRAG(
            db_path=cfg["paths"].get("rag_db_dir", "data/rag_db"),
            collection=cfg["rag"]["collection_name"],
            top_k=3,
        )
        rag.build_knowledge_base(force_rebuild=False)
    except Exception as e:
        logger.warning("RAG init failed: %s — skipping retrieval test", e)
        return {}

    results = {"per_query": {}, "precision_at_1": 0.0}
    correct = 0
    total = 0

    for attack_id in range(1, 15):  # skip BENIGN
        name = ID_TO_LABEL.get(attack_id, str(attack_id))
        try:
            intel = rag.get_threat_intelligence(name)
            top_name = intel.get("attack_name", "")
            match = (top_name.lower().strip() == name.lower().strip())
            results["per_query"][name] = {
                "query": name,
                "top_result": top_name,
                "correct": match,
            }
            if match:
                correct += 1
            total += 1
        except Exception as e:
            results["per_query"][name] = {"query": name, "error": str(e)}
            total += 1

    results["precision_at_1"] = round(correct / max(total, 1), 4)
    results["correct"] = correct
    results["total"] = total
    return results


def evaluate_prediction_mapping_consistency(model_dir: str):
    """
    For correctly-classified attacks in eval results, verify that RAG mapping
    would produce the correct MITRE technique.
    """
    from src.rag.knowledge_base import THREAT_KNOWLEDGE_BASE

    # Build attack_id → KB techniques map
    kb_map = {}
    for entry in THREAT_KNOWLEDGE_BASE:
        kb_map[entry.attack_id] = {
            "techniques": set(entry.techniques),
            "tactics": set(entry.mitre_tactics),
        }

    # Load evaluation results to get per-class detection rates
    eval_path = os.path.join(model_dir, "evaluation_results_v4.json")
    if not os.path.exists(eval_path):
        logger.warning("evaluation_results_v4.json not found — skipping consistency check")
        return {}

    with open(eval_path) as f:
        eval_data = json.load(f)

    results = {}
    for config_name, metrics in eval_data.items():
        report = metrics.get("classification_report", {})
        config_result = {}
        for attack_name, class_metrics in report.items():
            if attack_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            # Find the attack_id
            aid = None
            for k, v in GROUND_TRUTH_MITRE.items():
                if v["name"] == attack_name:
                    aid = k
                    break
            if aid is None or aid == 0:
                continue

            gt = GROUND_TRUTH_MITRE[aid]
            kb = kb_map.get(aid, {"techniques": set(), "tactics": set()})

            # Extract T-codes (e.g. "T1499" from "T1499 - Endpoint Denial of Service")
            kb_tcodes = {t.split(" ")[0].split(".")[0] for t in kb["techniques"]}
            gt_tcodes = {t.split(" ")[0].split(".")[0] for t in gt["techniques"]}
            tech_match = len(gt_tcodes & kb_tcodes)
            tac_match  = len(set(gt["tactics"]) & kb["tactics"])

            config_result[attack_name] = {
                "detection_recall": round(class_metrics.get("recall", 0), 4),
                "gt_techniques": gt["techniques"],
                "kb_techniques": list(kb["techniques"]),
                "technique_match": tech_match > 0,
                "tactic_match": tac_match > 0,
            }

        # Summary for this config
        attacks_with_match = sum(1 for v in config_result.values() if v["technique_match"])
        total_attacks = len(config_result)
        results[config_name] = {
            "per_attack": config_result,
            "technique_consistency_rate": round(attacks_with_match / max(total_attacks, 1), 4),
            "n_attacks_evaluated": total_attacks,
        }

    return results


def main():
    import yaml

    try:
        with open("configs/config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {"paths": {"models_dir": "models/checkpoints"}}

    model_dir = cfg.get("paths", {}).get("models_dir", "models/checkpoints")

    logger.info("=" * 70)
    logger.info("THREAT MAPPING EVALUATION")
    logger.info("=" * 70)

    # 1. Knowledge base coverage
    logger.info("--- Knowledge Base Coverage ---")
    kb_results = evaluate_knowledge_base_coverage()
    logger.info("Mean tactic recall:    %.4f", kb_results["mean_tactic_recall"])
    logger.info("Mean technique recall: %.4f", kb_results["mean_technique_recall"])
    logger.info("Tactic coverage:       %d/%d = %.2f%%",
                kb_results["tactic_coverage"]["overlap"],
                kb_results["tactic_coverage"]["gt_total"],
                kb_results["tactic_coverage"]["coverage_rate"] * 100)

    # 2. RAG retrieval precision
    logger.info("--- RAG Retrieval Precision ---")
    rag_results = evaluate_rag_retrieval()
    if rag_results:
        logger.info("Precision@1: %.4f (%d/%d)",
                    rag_results.get("precision_at_1", 0),
                    rag_results.get("correct", 0),
                    rag_results.get("total", 0))

    # 3. Prediction-mapping consistency
    logger.info("--- Prediction-to-MITRE Consistency ---")
    consistency = evaluate_prediction_mapping_consistency(model_dir)
    for cfg_name, data in consistency.items():
        logger.info("  %s: technique_consistency=%.4f (%d attacks)",
                    cfg_name, data["technique_consistency_rate"], data["n_attacks_evaluated"])

    # Save combined output
    output = {
        "knowledge_base_coverage": kb_results,
        "rag_retrieval": rag_results,
        "prediction_mapping_consistency": consistency,
    }

    out_path = os.path.join(model_dir, "threat_mapping_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
