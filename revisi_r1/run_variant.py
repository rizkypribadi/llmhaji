"""Harness varian untuk Revisi R1 (E3, E7, A0 ulang).

Pipeline = ragasv2/eval_ragas.run_rag_pipeline, satu faktor diubah lewat flag:
    --prompt  strict | unconstrained | production | norag
    --rerank  on | off          (off = 5 teratas dense dari 25 kandidat yang sama)
    --verify  off | advisory | enforced
    --collection                default mutawwif_v4

Resume: jawaban di-cache per soal (answers.jsonl), skor RAGAS per soal (ragas.jsonl).
Keluaran di revisi_r1/results/<name>/:
    rag_run.csv           skema = rag_run_20260814_101304.csv
    ragas_results.json    skema = ragas_results_v4_deepseek.json (+ kolom tambahan di answers.jsonl)
    config_snapshot.json, run.log

Contoh:
    .venv\\Scripts\\python.exe -u revisi_r1\\run_variant.py --name A0_rerun
    .venv\\Scripts\\python.exe -u revisi_r1\\run_variant.py --name A1_unconstrained --prompt unconstrained --limit 10
"""
import argparse
import csv
import json
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from common import ROOT, RESULTS, METRICS, ENFORCED_FALLBACK, load_dataset, ragas_score_rows
import common

# A1: PROMPT_KETAT dengan TIGA batasan dihapus (hanya-dari-konteks, tolak-bila-tidak-cukup,
# setiap-kalimat-tertelusur). Larangan gaya (salam, doa, tawaran lanjutan, hikmah) dan GAYA
# dipertahankan agar efek gaya tidak tercampur. Frasa "yang tidak ada di konteks" pada larangan
# hikmah juga dihapus karena itu bagian dari batasan konteks.
PROMPT_UNCONSTRAINED = """Anda adalah sistem penjawab pertanyaan fiqih haji dan umrah.

        ATURAN:
        - Gunakan KONTEKS yang diberikan sebagai bahan rujukan.
        - DILARANG menulis salam, sapaan, pujian, doa, harapan, atau penutup.
        - DILARANG menawarkan pertanyaan lanjutan.
        - DILARANG menambahkan hikmah, motivasi, atau nasihat.

        GAYA:
        - Langsung ke jawaban pada kalimat pertama. Tanpa pembuka.
        - Ringkas dan padat.
        - Bahasa Indonesia baku. Teks biasa dan **bold** saja.
        """

# A5 (tanpa RAG): gaya sama, tanpa konteks sama sekali.
PROMPT_NORAG = """Anda adalah sistem penjawab pertanyaan fiqih haji dan umrah.

        ATURAN:
        - DILARANG menulis salam, sapaan, pujian, doa, harapan, atau penutup.
        - DILARANG menawarkan pertanyaan lanjutan.
        - DILARANG menambahkan hikmah, motivasi, atau nasihat.

        GAYA:
        - Langsung ke jawaban pada kalimat pertama. Tanpa pembuka.
        - Ringkas dan padat.
        - Bahasa Indonesia baku. Teks biasa dan **bold** saja.
        """


def system_prompt(kind):
    from chat_engine_v2 import PROMPT_KETAT, PROMPT_RAMAH
    return {"strict": PROMPT_KETAT, "production": PROMPT_RAMAH,
            "unconstrained": PROMPT_UNCONSTRAINED, "norag": PROMPT_NORAG}[kind]


def user_message(query, context, kind):
    if kind == "norag":
        return f"""PERTANYAAN JAMAAH:
                    {query}

                    JAWABAN:"""
    # Salinan byte-per-byte dari chat_engine_v2.generate_response
    return f"""KONTEKS:
                    {context}

                    PERTANYAAN JAMAAH:
                    {query}

                    JAWABAN:"""


def generate(client, query, context, kind, model):
    """Parameter request identik dengan chat_engine_v2.generate_response(stream=True).
    Tambahan: menangkap finish_reason (untuk diagnosis jawaban kosong, E7)."""
    from chat_engine_v2 import ANSWER_MAX_TOKENS
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt(kind)},
                  {"role": "user", "content": user_message(query, context, kind)}],
        max_tokens=ANSWER_MAX_TOKENS, temperature=0.0, top_p=0.1, stream=True)
    parts, finish, n_chunks, n_reason = [], None, 0, 0
    for ch in stream:
        n_chunks += 1
        if not ch.choices:
            continue
        c = ch.choices[0]
        if c.delta.content is not None:
            parts.append(c.delta.content)
        if getattr(c.delta, "reasoning_content", None):
            n_reason += 1
        if c.finish_reason:
            finish = c.finish_reason
    return "".join(parts), {"finish_reason": finish, "stream_chunks": n_chunks, "reasoning_chunks": n_reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--collection", default="mutawwif_v4")
    ap.add_argument("--prompt", default="strict", choices=["strict", "unconstrained", "production", "norag"])
    ap.add_argument("--rerank", default="on", choices=["on", "off"])
    ap.add_argument("--verify", default="off", choices=["off", "advisory", "enforced"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="nomor soal 1-based, dipisah koma (mis. 84,113)")
    ap.add_argument("--workers", type=int, default=4, help="paralel generasi")
    ap.add_argument("--judge-workers", type=int, default=None, help="default = 3 seperti run final")
    ap.add_argument("--empty-retries", type=int, default=2,
                    help="ulangi generasi bila jawaban kosong (run final: 0)")
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge", default=common.R1_JUDGE, help="model judge RAGAS")
    ap.add_argument("--metrics", default=None,
                    help="subset metrik, mis. faithfulness,answer_relevancy (A1/A4: metrik konteks "
                         "tidak bergantung pada jawaban, jadi identik dengan A0 secara definisi)")
    a = ap.parse_args()

    out = RESULTS / a.name
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "run.log").open("a", encoding="utf-8")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    from config import OPENAI_MODEL, RETRIEVE_K, RERANK_K, EMBEDDING_MODEL, RERANKER_MODEL, NLI_MODEL
    model = a.model or OPENAI_MODEL
    questions, refs, _ = load_dataset()
    idx = list(range(len(questions)))
    if a.only:
        idx = [int(x) - 1 for x in a.only.split(",")]
    elif a.limit:
        idx = idx[:a.limit]

    snap = dict(name=a.name, started=datetime.now().isoformat(), args=vars(a),
                generator=dict(model=model, temperature=0.0, top_p=0.1, max_tokens=8192, seed=None,
                               stream=True, gateway="sumopod", account=common.DEFAULT_PROVIDER),
                retrieval=dict(collection=a.collection, embedding=EMBEDDING_MODEL, retrieve_k=RETRIEVE_K,
                               rerank_k=RERANK_K, reranker=RERANKER_MODEL, query_expansion="mati"),
                verify=dict(mode=a.verify, nli_model=NLI_MODEL, overlap_embedding=EMBEDDING_MODEL,
                            fallback=ENFORCED_FALLBACK if a.verify == "enforced" else None),
                system_prompt=system_prompt(a.prompt),
                judge=dict(model=a.judge, temperature=common.JUDGE_TEMPERATURE,
                           max_tokens=common.JUDGE_MAX_TOKENS, max_retries=common.JUDGE_MAX_RETRIES,
                           run_config=common.RUNCFG, answer_relevancy_strictness=common.ANSWER_STRICTNESS,
                           seed=None),
                questions=[i + 1 for i in idx], python=platform.python_version())
    import ragas
    snap["ragas_version"] = ragas.__version__
    (out / "config_snapshot.json").write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"=== {a.name}: prompt={a.prompt} rerank={a.rerank} verify={a.verify} coll={a.collection} n={len(idx)}")

    # ---- komponen ----
    from chat_engine_v2 import get_reranker
    client = common.build_chat_client()
    vs = reranker = None
    if a.prompt != "norag":
        from vector_manager import get_another_vector_store
        vs = get_another_vector_store(collection_name=a.collection)
        reranker = get_reranker() if a.rerank == "on" else None
    checker = None
    if a.verify != "off":
        from hallucination_checker import HallucinationChecker
        checker = HallucinationChecker(nli_model_name=NLI_MODEL, embedding_model_name=EMBEDDING_MODEL)
    local_lock = threading.Lock()  # model lokal (embedding/reranker/NLI) dipakai bergantian

    # ---- generasi (resume) ----
    cache = out / "answers.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); done[r["idx"]] = r
    todo = [i for i in idx if i not in done]
    log(f"[gen] {len(idx) - len(todo)} di cache, {len(todo)} perlu dibangkitkan")
    wlock = threading.Lock()

    def work(i):
        q = questions[i]
        t0 = time.time()
        rec = {"idx": i, "no": i + 1, "question": q}
        if vs is not None:
            with local_lock:
                docs = vs.as_retriever(search_kwargs={"k": RETRIEVE_K}).invoke(q)
                if reranker is not None:
                    sc = reranker.predict([(q, d.page_content) for d in docs])
                    ranked = sorted(zip(docs, sc), key=lambda x: x[1], reverse=True)[:RERANK_K]
                    rec["rerank_scores"] = [float(s) for _, s in ranked]
                    top = [d for d, _ in ranked]
                else:
                    top = docs[:RERANK_K]
            rec["contexts"] = [d.page_content for d in top]
            rec["sources"] = [d.metadata.get("source", "") for d in top]
            rec["chunk_ids"] = [d.metadata.get("_id", "") for d in top]
        else:
            rec["contexts"] = []
        ctx = "\n\n".join(rec["contexts"])
        attempts = []
        for k in range(a.empty_retries + 1):
            ans, meta = generate(client, q, ctx, a.prompt, model)
            attempts.append({**meta, "len": len(ans)})
            if ans.strip():
                break
        rec["raw_answer"] = ans
        rec["gen_attempts"] = attempts
        rec["answer"] = ans
        if checker is not None and ans.strip():
            with local_lock:
                v = checker.verify_answer(answer=ans, context=ctx, query=q)
            rec["verification"] = {k: (bool(x) if isinstance(x, (bool,)) else x) for k, x in v.items()}
            rec["verification"]["is_safe_fallback_rule"] = checker._is_safe_fallback(ans)
            if a.verify == "enforced" and not v["is_valid"]:
                rec["answer"] = ENFORCED_FALLBACK
                rec["blocked"] = True
        rec["seconds"] = round(time.time() - t0, 1)
        with wlock:
            with cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        return rec

    t_gen = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, i) for i in todo]
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
                done[r["idx"]] = r
                log(f"[gen] {n}/{len(todo)} no={r['no']} {r['seconds']}s len={len(r['answer'])} "
                    f"finish={r['gen_attempts'][-1]['finish_reason']} tries={len(r['gen_attempts'])}"
                    + (" BLOCKED" if r.get("blocked") else ""))
            except Exception as e:
                log(f"[gen] ERROR {type(e).__name__}: {e}")
    log(f"[gen] waktu generasi {time.time() - t_gen:.0f}s")
    missing = [i for i in idx if i not in done]
    if missing:
        log(f"[!] {len(missing)} soal gagal; jalankan ulang untuk resume. Berhenti.")
        return

    recs = [done[i] for i in idx]
    with (out / "rag_run.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        mx = max(len(r["contexts"]) for r in recs)
        w.writerow(["question", "answer"] + [f"context_{j + 1}" for j in range(mx)])
        for r in recs:
            w.writerow([r["question"], r["answer"]] + r["contexts"] + [""] * (mx - len(r["contexts"])))

    if a.no_ragas:
        log("[ragas] dilewati (--no-ragas)")
        return
    rows = [dict(user_input=r["question"], retrieved_contexts=r["contexts"], response=r["answer"],
                 reference=refs[r["idx"]]) for r in recs]
    names = METRICS if a.prompt != "norag" else ["answer_relevancy"]
    if a.metrics:
        names = a.metrics.split(",")
    t_j = time.time()
    scores = ragas_score_rows(rows, out / "ragas.jsonl", metric_names=names, label=a.name,
                              judge=common.build_judge(a.judge), judge_workers=a.judge_workers, log=log)
    log(f"[ragas] waktu judge {time.time() - t_j:.0f}s")
    final = []
    for row, s in zip(rows, scores):
        final.append({**row, **{m: s.get(m) for m in names}})
    (out / "ragas_results.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    for m in names:
        v = [x[m] for x in final if x[m] is not None]
        log(f"  {m:<24} n={len(v):>3} mean={sum(v) / max(len(v), 1):.4f}")
    log("selesai")


if __name__ == "__main__":
    main()
