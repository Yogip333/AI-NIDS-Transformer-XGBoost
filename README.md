# AI-NIDS: Self-Supervised Multi-View Transformer + XGBoost for Cross-Protocol Threat Detection

MSc Cyber Security dissertation project - University of Hertfordshire, 2026.

An end-to-end, session-level network intrusion detection system that groups traffic into 20-flow sessions, enriches CICIDS2017 flow records with a Zeek-informed 81-column feature schema (conn, DNS, HTTP, SSL and file-transfer views), learns traffic representations with a self-supervised Transformer, and classifies threats with XGBoost - served live over FastAPI/WebSocket with SHAP explainability and MITRE ATT&CK enrichment.

## Headline results

| Metric | Value |
|---|---|
| Held-out accuracy (CICIDS2017, multi-class) | 91.01% |
| Weighted F1 | 91.01% |
| AUC-ROC | 0.9876 |
| Streaming evaluation | 51,130 sessions / 1,022,600 flows |
| Attack recall (streaming) | 99.4% |
| Streaming detection accuracy | 90.8% |
| Streaming false-positive rate | 27.8% (analysed; calibration + hard-negative mining proposed) |

Learned session embeddings lift held-out accuracy from 61.93% (handcrafted statistics alone) to 91.01% - evidence that cross-protocol session context carries signal that per-flow statistics miss.

## Architecture

**Sessioniser** groups consecutive flows per source host into fixed-length 20-flow sessions (ablation confirmed 20 as the optimal window).

**Zeek-informed schema** extends CICIDS2017 records to 81 columns, mapping each flow to a log family (conn / dns / http / ssl / files) plus protocol, connection-state and service indicators.

**Self-supervised Transformer encoder** with 4 layers, 8 attention heads and hidden size 256, pre-trained with masked-feature modelling on 79,558 benign sessions (15 epochs), producing a 256-dimensional CLS session embedding.

**XGBoost classifier** performs multi-class threat classification on the learned embeddings.

**Serving and explainability:** FastAPI + WebSocket streaming inference, SQLite alert store, SHAP feature attribution, and a ChromaDB/RAG layer that enriches detections with curated MITRE ATT&CK tactics, techniques and analyst-facing summaries.

## Dataset

Uses CICIDS2017 (Sharafaldin, Lashkari & Ghorbani, 2018). The dataset is not included in this repository. Download it from the official source at https://www.unb.ca/cic/datasets/ids-2017.html and point the config at your local copy.

## Honest limitations

The 27.8% streaming false-positive rate is documented and analysed rather than hidden: benign bursts inflate false positives (4,498 FPs vs 210 missed attacks, a deliberate recall-first trade-off, since a missed attack is unobserved compromise while a false positive costs analyst minutes). Proposed fixes: probability calibration and hard-negative mining on burst traffic. Macro-F1 (60.24%) trails weighted F1 due to class imbalance in rare attack categories.

## Author

Yogisha Paneru - Cybersecurity Analyst @ CoreDefense | CEH | MSc Cyber Security

LinkedIn: https://www.linkedin.com/in/yogisha-paneru-5b592a16b/


## Copyright

Copyright (c) 2026 Yogisha Paneru. All rights reserved. This repository is shared publicly for portfolio and recruitment review. No permission is granted to copy, modify, redistribute, or present this work or its results as your own without the author's written consent. The dissertation this work is based on was submitted to the University of Hertfordshire.
