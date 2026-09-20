# run_llama3_always_rag.py
# -*- coding: utf-8 -*-
"""
اجرای Always-RAG برای llama3.1 (بدون gate)
تا Table 2 کامل شود
"""
import json
import os
import re
import time
import requests
from sentence_transformers import CrossEncoder

OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"
OLLAMA_EMB = "http://127.0.0.1:11434/api/embeddings"
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "family_medicine_engine"
MODEL = "llama3.1:8b"

TEMPERATURE = 0.0
MAX_TOKENS = 512
RETRIEVAL_LIMIT = 20
VECTOR_THRESHOLD = 0.25
TOP_K = 3
CHUNK_FLOOR = 0.50

OUTPUT_FILE = "results_llama3_always_rag.json"

print("⏳ بارگذاری Reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
print("✅ آماده.\n")

def ask_llama(prompt):
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_CHAT, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
            }, timeout=300)
            if r.ok:
                return r.json()["message"]["content"].strip()
        except Exception as e:
            print(f"   ⚠️ retry {attempt+1}: {e}")
            time.sleep(2)
    return ""

def extract_letter(text):
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    tail = "\n".join(lines[-5:])
    m = re.search(r"(?:FINAL ANSWER|answer|correct|choice)\s*(?:is|:|\-)?\s*[\(\[]?\s*([A-D])\b",
                  tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for line in reversed(lines[-5:]):
        m = re.fullmatch(r"[\(\[]?\s*([A-D])\s*[\)\]]?\.?", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.findall(r"\b([A-D])\b", tail)
    if m:
        return m[-1].upper()
    return None

def retrieve_and_rerank(question, options):
    try:
        emb = requests.post(OLLAMA_EMB,
            json={"model": "bge-m3", "prompt": question}, timeout=120
        ).json().get("embedding")
        if not emb:
            return [], 0.0
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            json={"vector": emb, "limit": RETRIEVAL_LIMIT,
                  "with_payload": True, "score_threshold": VECTOR_THRESHOLD},
            timeout=60)
        hits = r.json().get("result", []) if r.ok else []
        if not hits:
            return [], 0.0
        query = question + " " + " ".join(o[:80] for o in options)
        pairs = [[query, (h.get("payload", {}).get("full_text") or
                          h.get("payload", {}).get("text") or "")[:512]] for h in hits]
        scores = reranker.predict(pairs, batch_size=8)
        ranked = sorted(zip(hits, scores), key=lambda x: -x[1])
        top = [h for h, s in ranked[:TOP_K] if s >= CHUNK_FLOOR]
        top_score = float(ranked[0][1]) if ranked else 0.0
        return top, top_score
    except Exception as e:
        print(f"   ❌ retrieve: {e}")
        return [], 0.0

def build_rag_prompt(q, options, ctx):
    stem = (f"Answer the following medical question.\n\n{q}\n\n"
            f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\n"
            f"IMPORTANT: On the FINAL line, output ONLY the correct letter (A, B, C, or D).")
    return (f"Answer the following medical question.\n\n"
            f"=== RETRIEVED MEDICAL REFERENCE (for context only) ===\n{ctx}\n"
            f"=== END OF REFERENCE ===\n\n{stem}\n\n"
            f"=== CRITICAL INSTRUCTIONS ===\n"
            f"1. If the reference does NOT explicitly address the question, IGNORE it.\n"
            f"2. If the reference contradicts your knowledge, TRUST YOUR OWN KNOWLEDGE.\n"
            f"3. When in doubt, rely on your own medical training.")

# بارگذاری
print("📥 بارگذاری دیتاست...")
from datasets import load_dataset
ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

with open("test_ids.json", "r") as f:
    test_ids = json.load(f)
print(f"✅ {len(test_ids)} سوال test بارگذاری شد.\n")

# Resume
results = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
done_ids = {r["id"] for r in results}
print(f"♻️ {len(done_ids)} سوال قبلاً پردازش شده.\n")

remaining = [qid for qid in test_ids if qid not in done_ids]
print(f"🎯 باید {len(remaining)} سوال پردازش شود.\n")

start_time = time.time()

for i, qid in enumerate(remaining, 1):
    item = ds[qid]
    question = item["question"]
    options = [item["options"][k] for k in sorted(item["options"].keys())]
    correct = next((k for k, v in item["options"].items() if v == item["answer"]), None)

    print(f"\n--- [{i}/{len(remaining)}] ID {qid} ---")

    # Always-RAG: همیشه context استفاده می‌شود (بدون gate)
    chunks, top_score = retrieve_and_rerank(question, options)
    if chunks:
        ctx = "\n\n".join(
            (c.get("payload", {}).get("full_text") or
             c.get("payload", {}).get("text") or "")[:1200] for c in chunks
        )
        rag_ans = extract_letter(ask_llama(build_rag_prompt(question, options, ctx)))
    else:
        rag_ans = None
    print(f"  🟥 Always-RAG (score={top_score:.3f}): {rag_ans}")

    rec = {
        "id": qid,
        "correct": correct,
        "rag_always": rag_ans,
        "top_score": top_score,
        "has_context": bool(chunks),
    }
    results.append(rec)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if i % 10 == 0:
        elapsed = time.time() - start_time
        avg = elapsed / i
        rem = avg * (len(remaining) - i)
        print(f"\n⏱️ {i}/{len(remaining)} | سپری: {elapsed/60:.1f} دقیقه | باقی: {rem/60:.1f} دقیقه\n")

print(f"\n✅ تمام شد! ذخیره در {OUTPUT_FILE}")

# خلاصه
n = len(results)
acc = sum(1 for r in results if r["rag_always"] == r["correct"])
print(f"\n📊 Always-RAG accuracy: {acc}/{n} = {100*acc/n:.1f}%")