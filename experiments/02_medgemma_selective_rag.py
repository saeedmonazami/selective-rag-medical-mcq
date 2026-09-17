# ============================================================
# 2_run_medgemma_final.py — نسخهٔ نهایی ادغام‌شده
#  • استخراج حرف پاسخ از ۳ خط آخر (الگوی FINAL ANSWER)
#  • بلوک CRITICAL INSTRUCTIONS ضد مسمومیت (Context Poisoning)
#  • شمارنده‌های کامل: rag_used / none / helped / hurt / منابع
#  • کالیبراسیون آستانه فقط روی Validation، گزارش فقط روی Test
#  • کش و ادامه‌پذیری (Resume)
# ============================================================
import json, math, random, re, time, requests
from collections import Counter
from datasets import load_dataset
from sentence_transformers import CrossEncoder

# ---------------------------- تنظیمات ----------------------------
API_URL   = "http://127.0.0.1:1234/v1/chat/completions"   # LM Studio
MODEL     = "medgemma-1.5-4b-it"
EMB_URL   = "http://127.0.0.1:11434/api/embeddings"       # Ollama
EMB_MODEL = "bge-m3"
QDRANT    = "http://127.0.0.1:6333"
COLLECTION= "family_medicine_engine"

VAL_SIZE, TEST_SIZE = 100, 150     # 0 = همه
SEED              = 42
RETRIEVAL_LIMIT   = 20             # بازیابی وسیع برای recall
VECTOR_THRESHOLD  = 0.25           # آستانهٔ برداری (فیلتر اصلی: reranker)
TOP_K             = 3              # نکته: با ۱، کم‌نویزترین حالت (مثل بنچمارک ۳۰۰تایی)
CHUNK_FLOOR       = 0.50           # چانک زیر این امتیاز هرگز وارد پرامپت نمی‌شود
PRIORITY_BOOST    = 0.05           # وزن متادیتای priority در فضای احتمال
RERANKER_NAME     = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE   = "cpu"          # اگر VRAM خالی دارید: "cuda"
THRESHOLDS        = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_TOKENS        = 2048
TEMPERATURE       = 0.0
CACHE_FILE        = "results_medgemma_final.json"
SUMMARY_MD        = "summary_medgemma_final.md"
SUMMARY_JSON      = "summary_medgemma_final.json"

# ---------------------------- توابع کمکی ----------------------------
def ask(prompt):
    for attempt in range(3):
        try:
            r = requests.post(API_URL, json={
                "model": MODEL, "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}]}, timeout=300)
            if r.ok:
                return (r.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            print(f"  ⚠️ MedGemma error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return ""

def extract_letter(text):
    """استخراج حرف پاسخ — فقط ۳ خط آخر بررسی می‌شود (نسخهٔ جنگ‌آزموده)."""
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    tail = "\n".join(lines[-3:])
    m = re.search(r"(?:FINAL ANSWER|answer|پاسخ|correct answer|correct option|choice)"
                  r"\s*(?:is|:|\-)?\s*[\(\[]?\s*([A-D])\b", tail, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for line in reversed(lines[-3:]):
        m = re.fullmatch(r"[\(\[]?\s*([A-D])\s*[\)\]]?\.?", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.findall(r"\b([A-D])\b", tail)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([A-D])\b", text)
    return m[-1].upper() if m else None

def _to_probs(scores):
    """اگر reranker خروجی logit داد به احتمال تبدیل کن، وگرنه همان احتمال است."""
    scores = [float(s) for s in scores]
    if any(s < -0.001 or s > 1.001 for s in scores):
        scores = [1.0 / (1.0 + math.exp(-s)) for s in scores]
    return scores

def retrieve_and_rerank(question, options):
    """بازیابی وسیع → بازنمره‌دهی → تقویت با priority → مرتب‌سازی."""
    try:
        emb = requests.post(EMB_URL, json={"model": EMB_MODEL, "prompt": question},
                            timeout=120).json().get("embedding")
        if not emb:
            return []
        r = requests.post(f"{QDRANT}/collections/{COLLECTION}/points/search",
                          json={"vector": emb, "limit": RETRIEVAL_LIMIT,
                                "with_payload": True,
                                "score_threshold": VECTOR_THRESHOLD}, timeout=60)
        hits = r.json().get("result", []) if r.ok else []
        if not hits:
            return []
        query = question + " " + " ".join(o[:80] for o in options)
        texts = [(h.get("payload", {}).get("full_text") or
                  h.get("payload", {}).get("text") or "")[:512] for h in hits]
        probs = _to_probs(reranker.predict([[query, t] for t in texts], batch_size=16))
        out = []
        for h, p in zip(hits, probs):
            pay = h.get("payload", {}) or {}
            boost = PRIORITY_BOOST * ((pay.get("priority", 80) or 80) - 80) / 15.0
            out.append({"score": min(max(p + boost, 0.0), 1.0), "payload": pay})
        out.sort(key=lambda x: -x["score"])
        return out
    except Exception as e:
        print(f"  ❌ retrieval error: {e}")
        return []

def build_context(ranked):
    """ساخت متن مرجع از چانک‌های بالای کف امتیاز؛ بدون هیچ خط اضافی."""
    kept = [c for c in ranked[:TOP_K] if c["score"] >= CHUNK_FLOOR]
    if not kept:
        return "", 0.0, []
    parts, srcs = [], []
    for i, c in enumerate(kept, 1):
        p = c["payload"]
        txt = (p.get("full_text") or p.get("text") or "")[:1200]
        parts.append(f"[Ref {i} (score={c['score']:.3f}): {p.get('source','Unknown')}]\n{txt}")
        srcs.append(p.get("source", "Unknown"))
    return "\n\n".join(parts), kept[0]["score"], srcs

def local_prompt(q, opts):
    return (f"Answer the following medical question.\n\n{q}\n\n"
            f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n\n"
            f"IMPORTANT: On the FINAL line, output ONLY the correct letter (A, B, C, or D).")

def rag_prompt(q, opts, ctx):
    return (f"Answer the following medical question.\n\n"
            f"=== RETRIEVED MEDICAL REFERENCE (for context only) ===\n{ctx}\n"
            f"=== END OF REFERENCE ===\n\nQuestion:\n{q}\n\n"
            f"A) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\n\n"
            f"=== CRITICAL INSTRUCTIONS ===\n"
            f"1. The reference above was retrieved by semantic similarity and may NOT "
            f"directly answer this specific question.\n"
            f"2. If the reference does NOT explicitly and directly address this exact "
            f"clinical scenario, IGNORE it completely.\n"
            f"3. If the reference contradicts your own medical knowledge, TRUST YOUR "
            f"OWN KNOWLEDGE and IGNORE the reference.\n"
            f"4. Only use the reference if it provides the EXACT answer for this "
            f"specific case.\n"
            f"5. When in doubt, rely on your own medical training.\n\n"
            f"IMPORTANT: On the FINAL line, output ONLY the correct letter (A, B, C, or D).")

def wilson_ci(correct, total):
    if total == 0:
        return (0.0, 0.0)
    z, p = 1.96, correct / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    m = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (100 * (c - m), 100 * (c + m))

def mcnemar(rows, key_final):
    b = sum(1 for r in rows if r["local"] == r["correct"] and r[key_final] != r["correct"])
    c = sum(1 for r in rows if r[key_final] == r["correct"] and r["local"] != r["correct"])
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    return b, c, chi2, math.erfc(math.sqrt(chi2 / 2))

# ---------------------------- بارگذاری ----------------------------
print("⏳ بارگذاری Reranker...")
reranker = CrossEncoder(RERANKER_NAME, device=RERANKER_DEVICE)
print("✅ آماده.\n")

print("📥 بارگذاری MedQA...")
ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
random.seed(SEED)
idx = list(range(len(ds)))
random.shuffle(idx)
val_idx  = set(idx[:VAL_SIZE]) if VAL_SIZE else set()
test_idx = idx[VAL_SIZE:VAL_SIZE + TEST_SIZE] if TEST_SIZE else idx[VAL_SIZE:]
items = [(i, ds[i]) for i in sorted(val_idx | set(test_idx))]
print(f"✅ Val: {len(val_idx)} | Test: {len(test_idx)}\n")

try:
    cache = {r["id"]: r for r in json.load(open(CACHE_FILE, encoding="utf-8"))}
except FileNotFoundError:
    cache = {}

# ---------------------------- مرحله ۱: محاسبهٔ پاسخ‌ها ----------------------------
print("🧠 مرحله ۱: محاسبهٔ Local و RAG-full (با کش)...")
t0 = time.time()
for n, (i, item) in enumerate(items, 1):
    if i in cache:
        continue
    q = item["question"]
    opts = [item["options"][k] for k in sorted(item["options"].keys())]
    correct = next((k for k, v in item["options"].items() if v == item["answer"]), None)

    loc = extract_letter(ask(local_prompt(q, opts)))
    ranked = retrieve_and_rerank(q, opts)
    ctx, top, srcs = build_context(ranked)
    rag = extract_letter(ask(rag_prompt(q, opts, ctx))) if ctx else None

    cache[i] = {"id": i, "correct": correct, "local": loc, "rag_full": rag,
                "top_score": top, "has_context": bool(ctx), "sources": srcs,
                "in_val": i in val_idx}
    json.dump(list(cache.values()), open(CACHE_FILE, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if n % 25 == 0:
        el = time.time() - t0
        print(f"  ⏱️ {n}/{len(items)} | سپری: {el/60:.1f} دقیقه | "
              f"تخمین باقی: {el/n*(len(items)-n)/60:.1f} دقیقه")

rows = list(cache.values())
val_rows  = [r for r in rows if r.get("in_val")]
test_rows = [r for r in rows if not r.get("in_val")]

# ---------------------------- مرحله ۲: کالیبراسیون روی Val ----------------------------
def gated_acc(rs, th):
    ok = used = 0
    for r in rs:
        use = r["has_context"] and r["rag_full"] is not None and r["top_score"] >= th
        fin = r["rag_full"] if use else r["local"]
        ok += fin == r["correct"]
        used += use
    return (ok / len(rs) if rs else 0.0), used

print("\n🔍 مرحله ۲: کالیبراسیون آستانه روی Validation...")
cands = []
for th in THRESHOLDS:
    acc, used = gated_acc(val_rows, th)
    cands.append({"th": th, "acc": acc, "used": used})
    print(f"   th={th:.2f} → val acc {acc*100:5.2f}% (RAG used {used}/{len(val_rows)})")
best = max(cands, key=lambda c: (c["acc"], c["th"]))   # در تساوی، آستانهٔ محافظه‌کارتر
print(f"✅ بهترین آستانه: {best['th']:.2f} (val acc {best['acc']*100:.2f}%)\n")

# ---------------------------- مرحله ۳: ارزیابی روی Test ----------------------------
def evaluate(rs, th, label):
    n = len(rs)
    lc = rc = ac = used_s = used_a = 0
    helped = hurt = 0
    none_loc = none_rag = none_fin = 0
    srcs = Counter()
    for r in rs:
        use_s = r["has_context"] and r["rag_full"] is not None and r["top_score"] >= th
        use_a = r["has_context"] and r["rag_full"] is not None
        fin_s = r["rag_full"] if use_s else r["local"]
        fin_a = r["rag_full"] if use_a else r["local"]
        used_s += use_s; used_a += use_a
        if use_s:
            srcs.update(r["sources"])
        lok = r["local"] == r["correct"]
        sok = fin_s == r["correct"]
        aok = fin_a == r["correct"]
        lc += lok; rc += sok; ac += aok
        if sok and not lok: helped += 1
        if lok and not sok: hurt += 1
        none_loc += r["local"] is None
        none_rag += (r["rag_full"] is None and r["has_context"])
        none_fin += fin_s is None
    ci_l, ci_s, ci_a = (wilson_ci(x, n) for x in (lc, rc, ac))
    b, c, chi2, p = mcnemar(rs, "_sel") if False else (None,)*4  # placeholder removed
    # McNemar برای بازوی انتخابی در برابر Local:
    b = sum(1 for r in rs if r["local"] == r["correct"] and
            (r["rag_full"] if (r["has_context"] and r["rag_full"] is not None
                               and r["top_score"] >= th) else r["local"]) != r["correct"])
    c = sum(1 for r in rs if (r["rag_full"] if (r["has_context"] and r["rag_full"] is not None
                              and r["top_score"] >= th) else r["local"]) == r["correct"]
            and r["local"] != r["correct"])
    if b + c:
        chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
        p = math.erfc(math.sqrt(chi2 / 2))
    else:
        chi2, p = 0.0, 1.0
    print("=" * 70)
    print(f"📊 {label} (n={n}, gate th={th:.2f})")
    print(f"   Local-only   : {lc}/{n} = {100*lc/n:5.2f}%  CI {ci_l[0]:.1f}-{ci_l[1]:.1f}")
    print(f"   Always-RAG   : {ac}/{n} = {100*ac/n:5.2f}%  CI {ci_a[0]:.1f}-{ci_a[1]:.1f} (used {used_a})")
    print(f"   Selective-RAG: {rc}/{n} = {100*rc/n:5.2f}%  CI {ci_s[0]:.1f}-{ci_s[1]:.1f} (used {used_s})")
    print(f"   McNemar (selective vs local): b={b} c={c} chi2={chi2:.2f} p={p:.4f}")
    print(f"   RAG helped {helped} | RAG hurt {hurt}")
    print(f"   None-counts: local={none_loc} rag_full={none_rag} final={none_fin}")
    if srcs:
        print("   Top sources:", ", ".join(f"{s}×{k}" for s, k in srcs.most_common(5)))
    print("=" * 70)
    return {"label": label, "n": n, "local": lc/n, "always": ac/n, "selective": rc/n,
            "used_selective": used_s, "used_always": used_a, "helped": helped,
            "hurt": hurt, "mcnemar": {"b": b, "c": c, "chi2": round(chi2, 2),
            "p": round(p, 4)}, "none": {"local": none_loc, "rag_full": none_rag,
            "final": none_fin}, "sources": srcs.most_common(5),
            "ci": {"local": ci_l, "always": ci_a, "selective": ci_s}}

print("\n🎯 مرحله ۳: ارزیابی نهایی روی Test...")
res = evaluate(test_rows, best["th"], "TEST — Selective vs Always vs Local")

# ---------------------------- مرحله ۴: ذخیرهٔ خلاصه ----------------------------
with open(SUMMARY_MD, "w", encoding="utf-8") as f:
    f.write(f"# MedGemma 1.5-4B — Selective RAG (final merged run)\n\n"
            f"- Best gate threshold (val): {best['th']:.2f} (val acc {best['acc']*100:.2f}%)\n"
            f"- Test Local-only : {res['local']*100:.2f}%\n"
            f"- Test Always-RAG : {res['always']*100:.2f}%\n"
            f"- Test Selective  : {res['selective']*100:.2f}% (used {res['used_selective']}/{res['n']})\n"
            f"- McNemar p (selective vs local): {res['mcnemar']['p']:.4f}\n"
            f"- RAG helped {res['helped']} / hurt {res['hurt']}\n"
            f"- None-counts: {res['none']}\n"
            f"- Top sources: {res['sources']}\n")
json.dump({"best_threshold": best["th"], "calibration": cands, "test": res},
          open(SUMMARY_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n✅ {SUMMARY_MD} و {SUMMARY_JSON} ذخیره شدند.")