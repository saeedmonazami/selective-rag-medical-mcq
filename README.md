# Selective RAG for Medical MCQs

Code and results for the paper:

> **Selective Retrieval-Augmented Generation for Medical MCQs: Context Poisoning Across Generic and Domain-Specific LLMs**
> Saeed Monazami Eslami, Ali Asghar Safaei

## Key Findings

### Qwen 35B (Generic MoE)
- Local accuracy: **64.2%** (817/1273)
- Paired Local vs Always-RAG (n=848): 63.4% vs **61.6%**
- McNemar: p=0.0014 (significant degradation)

### MedGemma 4B (Domain-specific)
- Local: **60.67%** (91/150)
- Always-RAG: **58.67%** (88/150)
- Selective-RAG (tau=0.90): **60.00%** (90/150, used 16/150)
- McNemar p (selective vs local): 1.0000

## Structure

- `experiments/` - Evaluation scripts
- `results/` - Output files
- `configs/` - Configuration files
- `docs/` - Methodology docs
- `supplementary/` - Prompts and methods

## License

MIT License
