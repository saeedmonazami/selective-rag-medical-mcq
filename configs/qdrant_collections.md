# Qdrant Collections

All collections are embedded with BAAI/bge-m3 (1024-dim vectors) via Ollama.

| Collection | Sources | Used by |
|-----------|---------|---------|
| family_medicine_engine | Harrison's + Family Medicine + MedMCQA + BiomixQA | Qwen, MedGemma |
| harrison_chunks | Harrison's Principles of Internal Medicine | Qwen |

## Point Metadata

Each indexed point carries:
- `source`: Book/article name
- `priority`: Priority score (0-100)
- `full_text`: Full chunk text
- `text`: Shortened text
- `red_flag`: Red flag tags

## Important

**MedQA-USMLE items were excluded from the retrieval corpus prior to indexing to prevent benchmark contamination.**

## Qdrant Connection
URL: http://127.0.0.1:6333

## Search Parameters

```json
{
  "vector": [embedding],
  "limit": 8,
  "with_payload": true,
  "score_threshold": 0.50
}