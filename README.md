# Selective RAG for Medical MCQs

Code and results for the paper:

> **Model-dependent context poisoning in retrieval-augmented generation for medical multiple-choice questions**
> Saeed Monazami Eslami, Ali Asghar Safaei
> Tarbiat Modares University, Tehran, Iran

---

## Key Findings

We evaluated three LLMs on the MedQA-USMLE benchmark under three retrieval conditions: **Local** (no RAG), **Always-RAG**, and **Selective-RAG**.

**The effect of Always-RAG is model-dependent.**

| Model | Size | Type | Local | Always-RAG | Selective-RAG | Delta | McNemar p | n |
|-------|------|------|-------|------------|---------------|---|---|---|
| Qwen 3.6-35B | 35B MoE | Generic | 63.4% | **61.6%** | — | −1.8 pp | **0.0014** | 848 |
| MedGemma 1.5-4B | 4B | Domain | 60.7% | **58.7%** | 60.0% | −2.0 pp | n.s. | 150 |
| Llama 3.1 8B | 8B | Generic | 56.7% | **59.3%** | 56.7% | +2.7 pp | n.s. | 150 |

### Secondary Findings

- Reranker scores do **not** reliably distinguish helpful from harmful retrieval.
- **Knowledge Conflict** is the dominant failure mode (63% Qwen, 100% MedGemma, 100% Llama 3.1).
- A gate calibrated on one model (MedGemma, τ = 0.90) does **not** transfer to another model.

---

## Structure
.
├── configs/ # Model-specific configuration details
│ ├── qwen_config.yaml
│ ├── medgemma_config.yaml
│ ├── llama_config.md
│ └── qdrant_collections.md
├── experiments/ # Evaluation scripts
│ ├── 01_qwen_local_vs_rag.py
│ ├── 02_medgemma_selective_rag.py
│ ├── 03_llama3_selective_rag.py
│ └── 04_llama3_always_rag.py
├── results/ # Raw JSON and summary results
│ ├── qwen/
│ │ ├── results2.json
│ │ ├── medqa_summary.md
│ │ └── summary_v2.md
│ ├── medgemma/
│ │ ├── results_medgemma_final.json
│ │ ├── summary_medgemma_final.json
│ │ └── summary_medgemma_final.md
│ └── llama3/
│ ├── results_llama3_final.json
│ └── results_llama3_always_rag.json
├── figures/ # Publication-ready figures
│ ├── Figure1_architecture.png
│ ├── Figure2_accuracy_three_models.png
│ └── Figure3_mcnemar_qwen.png
├── docs/ # Methodology and reproducibility
├── supplementary/ # Prompt templates and methods
├── LICENSE
├── README.md
└── requirements.txt


---

## Models and Inference Infrastructure

| Model | Parameters | Backend | Quantization |
|-------|-----------|---------|--------------|
| MedGemma-1.5-4B-IT | 4B dense | LM Studio (:1234) | None |
| Qwen3.6-35B-A3B | 35B MoE | OpenAI-compat (:1919) | NVFP4 (4-bit) |
| Llama 3.1 8B | 8B | Ollama (:11434) | Q4_K_M |

All models were executed locally on an **RTX 4070 Laptop (8 GB VRAM, 32 GB RAM, CUDA 12.4, Ubuntu 24.04)**.

---

## Methods Summary

### Knowledge Base

Three Qdrant collections embedded with **BAAI/bge-m3** (1,024-dim, cosine):
- `family_medicine_engine` — Harrison's + Family Medicine 8th ed. + MedMCQA + BiomixQA
- `harrison_chunks` — Harrison's Principles
- `pubmedqa_reasoning` — PubMedQA reasoning traces

### Retrieval Settings

| Parameter | Qwen | MedGemma / Llama 3.1 |
|-----------|------|----------------------|
| Retrieval limit (L) | 8 | 20 |
| Vector threshold (tau_vec) | 0.50 | 0.25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | BAAI/bge-reranker-v2-m3 |
| Top-K | 3 | 3 |
| Chunk floor | — | 0.50 |
| Gate (tau_gate) | −1.5 (always-on) | 0.90 (calibrated) |

### Statistical Analysis

- **Wilson 95% CI** for each accuracy estimate
- **McNemar's paired test** with Edwards' continuity correction
- Significance level alpha = 0.05, uncorrected

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt


Run evaluations

python experiments/01_qwen_local_vs_rag.py
python experiments/02_medgemma_selective_rag.py
python experiments/03_llama3_selective_rag.py
python experiments/04_llama3_always_rag.py

Citation

@article{monazami2026selective,
  title={Model-dependent context poisoning in retrieval-augmented generation for medical multiple-choice questions},
  author={Monazami Eslami, Saeed and Safaei, Ali Asghar},
  year={2026}
}

License


MIT License - see LICENSE for details.

Contact
Saeed Monazami Eslami

Ali Asghar Safaei (corresponding) — aa.safaei@modares.ac.ir

