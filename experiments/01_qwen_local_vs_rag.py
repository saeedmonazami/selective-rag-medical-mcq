# ============================================================
# 2_run_final.py — نسخه نهایی با فیلتر هوشمند RAG
# ============================================================
import json, re, requests, time
from sentence_transformers import CrossEncoder
import numpy as np

# تنظیمات
API_URL = "http://127.0.0.1:1919/v1/chat/completions"
MODEL_NAME = "Qwen3.6-35B-A3B-NVFP4"
EMBEDDING_URL = "http://127.0.0.1:11434/api/embeddings"
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTIONS = ["family_medicine_engine", "harrison_chunks"]
GATE = 0.50
MAX_RETRIES = 2

print("⏳ بارگذاری Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("✅ آماده!")

def ask(messages):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(API_URL, json={
                "model": MODEL_NAME,
                "temperature": 0,
                "max_tokens": 128,
                "messages": messages,
                "chat_template_kwargs": {"enable_thinking": False}
            }, timeout=600)
            if r.ok:
                m = r.json()["choices"][0]["message"]
                return (m.get("content") or m.get("reasoning_content") or "").strip()
        except Exception as e:
            print(f"  تلاش {attempt+1}: {e}")
            time.sleep(2)
    return ""

def extract_letter(text):
    if not text: return None
    matches = re.findall(r"\b([A-D])\b", text)
    if matches: return matches[-1]
    matches = re.findall(r"[\(\"\']?([A-Da-d])[\)\"\']?", text)
    return matches[-1].upper() if matches else None

def rerank(query, chunks, top_k=3):
    if not chunks: return []
    pairs = [[query, (c.get("payload", {}).get("full_text") or c.get("payload", {}).get("text") or "")[:512]] for c in chunks]
    scores = reranker.predict(pairs)
    for i, c in enumerate(chunks):
        c["rerank_score"] = float(scores[i])
    return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]

def retrieve(question):
    try:
        emb = requests.post(EMBEDDING_URL, json={"model": "bge-m3", "prompt": question}, timeout=120).json().get("embedding")
        if not emb: return "", 0.0
        candidates = []
        for col in COLLECTIONS:
            r = requests.post(f"{QDRANT_URL}/collections/{col}/points/search", json={
                "vector": emb, "limit": 8, "with_payload": True, "score_threshold": GATE
            }, timeout=60)
            if r.ok:
                candidates.extend(r.json().get("result", []))
        if not candidates: return "", 0.0
        top = rerank(question, candidates, 3)
        context = "\n\n---\n\n".join((c.get("payload", {}).get("full_text") or c.get("payload", {}).get("text") or "")[:1200] for c in top)
        top_score = top[0].get("rerank_score", 0.0) if top else 0.0
        return context, top_score
    except Exception as e:
        print(f"  ❌ {e}")
        return "", 0.0

def build_prompt(q, context=None):
    stem = f"{q['question']}\nA) {q['a']}\nB) {q['b']}\nC) {q['c']}\nD) {q['d']}\nAnswer with ONLY the letter (A/B/C/D)."
    if not context: return stem
    return f"Medical references (use them ONLY if they directly address the question):\n\n{context}\n\n{stem}\n\nFirst, briefly explain reasoning. Then, output ONLY the letter (A/B/C/D)."

# اجرا
print("\n🚀 شروع تست نهایی (با فیلتر هوشمند)...")
questions = json.load(open("gmedqa_sample.json", "r", encoding="utf-8"))
print(f"📊 {len(questions)} سوال")

try:
    saved = json.load(open("results2.json", "r", encoding="utf-8"))
    done = {q["id"]: q for q in saved if q.get("done2")}
    print(f"♻️ {len(done)} سوال قبلاً پردازش شده")
except:
    done = {}

for idx, q in enumerate(questions, 1):
    if q["id"] in done:
        q.update(done[q["id"]])
        if idx % 50 == 0: print(f"⏩ {idx}/{len(questions)} رد شد")
        continue
    
    print(f"\n--- {idx}/{len(questions)} ---")
    
    # 1) Local
    local = extract_letter(ask([{"role": "user", "content": build_prompt(q)}]))
    q["local"] = local

    # 2) RAG
    ctx, score = retrieve(q["question"])
    q["top_score"] = score
    
    # 🔥 قانون جدید: فقط اگر score > -1.5 باشد از RAG استفاده کن
    if ctx and score > -1.5:
        rag_prompt = build_prompt(q, ctx)
        rag = extract_letter(ask([{"role": "user", "content": rag_prompt}]))
        print(f"✅ true:{q['correct']} | local:{local} | rag:{rag} | score:{score:.3f} (استفاده شد)")
    else:
        rag = local
        if ctx:
            print(f"⚠️ زمینه ضعیف (score:{score:.3f}) ← Fallback به Local")
        else:
            print(f"⚠️ بدون زمینه ← Fallback به Local")
        print(f"✅ true:{q['correct']} | local:{local} | rag:{rag} (Fallback)")

    q["rag"] = rag
    q["done2"] = True
    
    json.dump(questions, open("results2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("🎉 ALL DONE!")