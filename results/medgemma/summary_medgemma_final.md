# MedGemma 1.5-4B — Selective RAG (final merged run)

- Best gate threshold (val): 0.90 (val acc 74.00%)
- Test Local-only : 60.67%
- Test Always-RAG : 58.67%
- Test Selective  : 60.00% (used 16/150)
- McNemar p (selective vs local): 1.0000
- RAG helped 1 / hurt 2
- None-counts: {'local': 1, 'rag_full': 0, 'final': 1}
- Top sources: [('MedRAG Core Textbooks (Harrison, Nelson, etc.)', 42), ('MedMCQA', 1)]
