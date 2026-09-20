# run_llama3_final.py
# -*- coding: utf-8 -*-
"""
اجرای llama3.1:8b روی 150 سوال MedGemma test
همان pipeline: family_medicine_engine + CrossEncoder reranker + gate=0.90
"""
import json
import os
import re
import time
import requests
from sentence_transformers import CrossEncoder

# ============================================================
# تنظیمات (مثل MedGemma)
# ============================================================
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
GATE = 0.90

OUTPUT_FILE = "results_llama3_final.json"

# ============================================================
# بارگذاری Reranker (روی CPU برای صرفه‌جویی VRAM)
# ============================================================
print("⏳ بارگذاری Reranker (روی CPU)...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
print("✅ آماده.\n")

# ============================================================
# توابع
# ============================================================
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
    tail = "\n".join(lines[-3:])
    m = re.search(r"(?:FINAL ANSWER|answer|correct|choice)\s*(?:is|:|\-)?\s*[\(\[]?\s*([A-D])\b",
                  tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for line in reversed(lines[-3:]):
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
        # Rerank
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

def build_prompt(q, options, ctx=None):
    stem = (f"Answer the following medical question.\n\n{q}\n\n"
            f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\n"
            f"IMPORTANT: On the FINAL line, output ONLY the correct letter (A, B, C, or D).")
    if not ctx:
        return stem
    return (f"Answer the following medical question.\n\n"
            f"=== RETRIEVED MEDICAL REFERENCE (for context only) ===\n{ctx}\n"
            f"=== END OF REFERENCE ===\n\n{stem}\n\n"
            f"=== CRITICAL INSTRUCTIONS ===\n"
            f"1. If the reference does NOT explicitly address the question, IGNORE it.\n"
            f"2. If the reference contradicts your knowledge, TRUST YOUR OWN KNOWLEDGE.\n"
            f"3. When in doubt, rely on your own medical training.")

# ============================================================
# بارگذاری سوالات
# ============================================================
print("📥 بارگذاری دیتاست MedQA...")
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

# ============================================================
# اجرا
# ============================================================
remaining = [qid for qid in test_ids if qid not in done_ids]
print(f"🎯 باید {len(remaining)} سوال پردازش شود.\n")
print("=" * 70)

start_time = time.time()

for i, qid in enumerate(remaining, 1):
    item = ds[qid]
    question = item["question"]
    options = [item["options"][k] for k in sorted(item["options"].keys())]
    correct = next((k for k, v in item["options"].items() if v == item["answer"]), None)

    print(f"\n--- [{i}/{len(remaining)}] ID {qid} ---")

    # Local
    local_ans = extract_letter(ask_llama(build_prompt(question, options)))
    print(f"  🟦 Local: {local_ans}")

    # RAG
    chunks, top_score = retrieve_and_rerank(question, options)
    if chunks and top_score >= GATE:
        ctx = "\n\n".join(
            (c.get("payload", {}).get("full_text") or
             c.get("payload", {}).get("text") or "")[:1200] for c in chunks
        )
        rag_ans = extract_letter(ask_llama(build_prompt(question, options, ctx)))
        print(f"  🟥 RAG (score={top_score:.3f}, gate PASS): {rag_ans}")
    else:
        rag_ans = None
        print(f"  ⚠️ RAG gate FAIL (score={top_score:.3f}) → Local استفاده می‌شود")

    rec = {
        "id": qid,
        "correct": correct,
        "local": local_ans,
        "rag_full": rag_ans,
        "top_score": top_score,
        "has_context": bool(chunks),
    }
    results.append(rec)
    done_ids.add(qid)

    # ذخیره لحظه‌ای
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ETA
    if i % 10 == 0:
        elapsed = time.time() - start_time
        avg = elapsed / i
        remaining_time = avg * (len(remaining) - i)
        print(f"\n⏱️ پیشرفت: {i}/{len(remaining)} | سپری: {elapsed/60:.1f} دقیقه | "
              f"باقی: {remaining_time/60:.1f} دقیقه\n")

print("\n" + "=" * 70)
print(f"🎉 تمام شد! نتایج در {OUTPUT_FILE}")
print("=" * 70)

# خلاصه نهایی
n = len(results)
correct_local = sum(1 for r in results if r["local"] == r["correct"])
correct_rag = sum(1 for r in results
                  if (r["rag_full"] if r["rag_full"] else r["local"]) == r["correct"])
print(f"\n📊 خلاصه نهایی (n={n}):")
print(f"   Local accuracy: {correct_local}/{n} = {100*correct_local/n:.1f}%")
print(f"   RAG accuracy  : {correct_rag}/{n} = {100*correct_rag/n:.1f}%")