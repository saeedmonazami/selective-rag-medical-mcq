# Reproducibility Guide

## Environment

- **Hardware**: RTX 4070 Laptop (8 GB VRAM, Intel Core i7-13700H, 32 GB RAM)
- **OS**: Ubuntu 24.04
- **CUDA**: 12.4
- **Python**: 3.10+

## Required Services

| Service | URL | Purpose |
|---------|-----|---------|
| LM Studio | http://127.0.0.1:1234 | MedGemma inference |
| OpenAI-compatible | http://127.0.0.1:1919 | Qwen inference |
| Ollama | http://127.0.0.1:11434 | bge-m3 embeddings |
| Qdrant | http://127.0.0.1:6333 | Vector search |

## Setup

```bash
# 1. Install dependencies
pip install requests sentence-transformers datasets numpy openai

# 2. Download embedding model
ollama pull bge-m3

# 3. Load models
# - medgemma-1.5-4b-it in LM Studio -> port 1234
# - Qwen3.6-35B-A3B-NVFP4 via OpenAI-compatible endpoint -> port 1919

# 4. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 5. Build Qdrant collections (see configs/qdrant_collections.md)
Running Experiments
cd experiments

# Qwen evaluation (Local vs Always-RAG on 1,273 questions)
python 01_qwen_local_vs_rag.py

# MedGemma evaluation (Selective RAG on 250 questions: 100 val + 150 test)
python 02_medgemma_selective_rag.py
Expected Outputs
Script
Output Files
01_qwen_local_vs_rag.py
results2.json
02_medgemma_selective_rag.py
results_medgemma_final.json, summary_medgemma_final.md, summary_medgemma_final.json
Expected Results
Qwen 35B (n=1,273 full, n=848 paired)
Condition
Correct
Accuracy (95% CI)
Local (no RAG)
538/848
63.4% (60.1-66.6%)
Always-RAG
522/848
61.6% (58.2-64.8%)
McNemar: b=19, c=3, chi2=10.23, p=0.0014
MedGemma 4B (n=150 test)
Condition
Correct
Accuracy
Local-only
91/150
60.67%
Always-RAG
88/150
58.67%
Selective-RAG (tau=0.90, 16/150)
90/150
60.00%
McNemar (Selective vs Local): p=1.0000
Random Seed
All experiments use SEED = 42 for reproducibility of data splits.
Runtime Estimates
Experiment
Approximate Time
Qwen evaluation (1,273 questions)
~8 hours
MedGemma evaluation (250 questions)
~4 hours
