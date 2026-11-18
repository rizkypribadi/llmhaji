"""E1 / E2 — menilai ulang JAWABAN YANG SAMA dengan RAGAS, k kali.

Konfigurasi metrik, embedding, RunConfig = salinan eval_ragas.py (lihat common.py).
Tiap run disimpan di results/<name>/run<k>/ (ragas.jsonl resume + ragas_results.json).

    # E1: judge final, 5 kali
    python -u revisi_r1/rejudge.py --name E1_deepseek_pro --runs 5
    # E2: judge independen
    python -u revisi_r1/rejudge.py --name E2_gemini37flash --judge gemini/gemini-3.7-flash --runs 1
"""
import argparse
import json
import time
from datetime import datetime

import common
from common import BASELINE_JSON, METRICS, RESULTS, ragas_score_rows, build_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--input", default=str(BASELINE_JSON))
    ap.add_argument("--judge", default=common.JUDGE_MODEL)
    ap.add_argument("--provider", default=common.DEFAULT_PROVIDER)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--start-run", type=int, default=1)
    ap.add_argument("--metrics", default=",".join(METRICS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge-workers", type=int, default=None)
    a = ap.parse_args()

    rows = json.loads(open(a.input, encoding="utf-8").read())
    if a.limit:
        rows = rows[:a.limit]
    rows = [dict(user_input=r["user_input"], retrieved_contexts=r["retrieved_contexts"],
                 response=r["response"] or "", reference=r["reference"]) for r in rows]
    names = a.metrics.split(",")
    base = RESULTS / a.name
    base.mkdir(parents=True, exist_ok=True)
    logf = (base / "run.log").open("a", encoding="utf-8")

    def log(m):
        line = f"[{datetime.now():%H:%M:%S}] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    snap = dict(name=a.name, input=a.input, n=len(rows), judge=a.judge, provider=a.provider,
                temperature=common.JUDGE_TEMPERATURE, max_tokens=common.JUDGE_MAX_TOKENS,
                max_retries=common.JUDGE_MAX_RETRIES, run_config=common.RUNCFG,
                judge_workers=a.judge_workers, metrics=names,
                answer_relevancy_strictness=common.ANSWER_STRICTNESS, embedding="BAAI/bge-m3 (lokal)",
                seed=None, runs=a.runs, started=datetime.now().isoformat())
    import ragas
    snap["ragas_version"] = ragas.__version__
    (base / "config_snapshot.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")

    judge = build_judge(a.judge, a.provider)
    for k in range(a.start_run, a.start_run + a.runs):
        d = base / f"run{k}"
        d.mkdir(exist_ok=True)
        t0 = time.time(); tok0 = dict(common.TOKENS)
        log(f"=== run {k} judge={a.judge} n={len(rows)}")
        sc = ragas_score_rows(rows, d / "ragas.jsonl", judge=judge, metric_names=names,
                              label=f"{a.name}_run{k}", judge_workers=a.judge_workers, log=log)
        final = [{**r, **{m: s.get(m) for m in names}} for r, s in zip(rows, sc)]
        (d / "ragas_results.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        used = {x: common.TOKENS[x] - tok0[x] for x in common.TOKENS}
        log(f"run {k} selesai {time.time() - t0:.0f}s | token judge: {used}")
        for m in names:
            v = [x[m] for x in final if x[m] is not None]
            log(f"  {m:<24} n={len(v):>3} mean={sum(v) / max(len(v), 1):.4f}")


if __name__ == "__main__":
    main()
