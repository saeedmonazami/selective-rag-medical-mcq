# Methodology

## Study Design

We evaluated the effect of RAG on medical MCQ accuracy using a controlled paired design across two model families.

### Models

| Model | Type | Purpose |
|-------|------|---------|
| Qwen3.6-35B-A3B-NVFP4 | Generic 35B MoE | Tests if a large generic model benefits from RAG |
| MedGemma-1.5-4B-IT | Domain-specific 4B | Tests if medical pretraining protects against context poisoning |

### Experimental Conditions

| Condition | Description |
|-----------|-------------|
| Local | No retrieval; model answers from parametric knowledge only |
| Always-RAG | Retrieved context always included in prompt |
| Selective-RAG | Context included only when rerank score exceeds calibrated threshold |

## Retrieval Pipeline

For each question:
1. Encode question with BAAI/bge-m3 (via Ollama)
2. Search target collection(s) in Qdrant
3. Re-rank candidates with cross-encoder
4. Retain top-K chunks above score floor
5. Optionally gate on rerank score threshold

### Qwen Retrieval Configuration

| Parameter | Value |
|-----------|-------|
| Collections | family_medicine_engine, harrison_chunks |
| Retrieval limit | 8 |
| Vector score threshold | 0.50 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Top-K chunks | 3 |
| Context length per chunk | 1,200 chars |
| RAG gate | rerank score > -1.5 (effectively always-on) |

### MedGemma Retrieval Configuration

| Parameter | Value |
|-----------|-------|
| Collections | family_medicine_engine |
| Retrieval limit | 20 |
| Vector score threshold | 0.25 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Top-K chunks | 3 |
| Chunk score floor | 0.50 |
| Priority metadata boost | 0.05 |
| Context length per chunk | 1,200 chars |
| RAG gate | Calibrated (selected threshold = 0.90) |

**Note:** The two models used different retrieval configurations, reflecting practical constraints of each serving backend. This is acknowledged as a limitation.

## Inference Configuration

| Parameter | Qwen | MedGemma |
|-----------|------|----------|
| API endpoint | http://127.0.0.1:1919/v1/chat/completions | http://127.0.0.1:1234/v1/chat/completions |
| Temperature | 0 | 0.0 |
| Max tokens | 128 | 2048 |
| Quantization | NVFP4 (4-bit) | None |
| Serving backend | OpenAI-compatible | LM Studio |

## Dataset

- **MedQA-USMLE-4-options**: GBaker/MedQA-USMLE-4-options, split = test
- Total questions: 1,273
- Random seed: 42 (for all data splits)

### Data Splits

| Experiment | Split | Size |
|------------|-------|------|
| Qwen Local evaluation | Full test set | 1,273 |
| Qwen paired evaluation | Subset with valid answers | 848 |
| MedGemma validation | Held-out validation | 100 |
| MedGemma test | Held-out test | 150 |

## Selective RAG Calibration

1. Split data: 100 validation + 150 test (seed = 42)
2. Compute Local and RAG-full answers for all questions
3. For each threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
   - If rerank_score >= threshold AND context exists: use RAG answer
   - Otherwise: use Local answer
4. Select threshold maximizing validation accuracy
5. Apply selected threshold to test set (never used for tuning)

**Best threshold: 0.90 (validation accuracy: 74.00%)**

## Statistical Analysis

- **Wilson 95% CI** for accuracy estimates
- **McNemar's test** with Edwards' continuity correction:
  - chi2