# Llama 3.1 8B Configuration

- **Model ID**: llama3.1:8b
- **Backend**: Ollama (port 11434)
- **Quantization**: Q4_K_M
- **Temperature**: 0
- **Max tokens**: 512

## Retrieval

- Collection: family_medicine_engine
- Retrieval limit (L): 20
- Vector threshold: 0.25
- Reranker: BAAI/bge-reranker-v2-m3
- Top-K: 3
- Chunk floor: 0.50
- Gate (tau_gate): 0.90 (transferred from MedGemma, not re-calibrated)

## Evaluation

- Same 150 test questions as MedGemma
- Random seed: 42
- Conditions: Local, Always-RAG, Selective-RAG

## Results

### Selective-RAG (with MedGemma-calibrated gate tau=0.90)

- Local: 56.7% (85/150)
- Selective-RAG: 56.7% (85/150)
- Gate invoked in 11/150 cases (7.3%)

### Always-RAG

- Always-RAG: 59.3% (89/150)
- Delta vs Local: +2.7 pp (n.s., McNemar p=0.5563)
- RAG hurt: 11 cases
- RAG helped: 15 cases
