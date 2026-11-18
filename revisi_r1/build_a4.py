"""E3-A4 — NLI layer ENFORCED, dirakit dari A0_rerun (mode advisory).

A0 dijalankan dengan --verify advisory: jawaban tidak diubah, verdict modul asli
tersimpan per soal. A4 = jawaban A0, tetapi yang verdict-nya tidak valid diganti
pesan fallback (persis seperti --verify enforced). Retrieval dan konteks identik,
sehingga:
  - context_precision / context_recall / context_entity_recall = A0 (metrik ini tidak
    membaca 'response'; dicek dari _required_columns ragas 0.4.3)
  - faithfulness & answer_relevancy: soal yang TIDAK diblokir = skor A0 (jawaban sama);
    soal yang diblokir dinilai ulang dengan judge yang sama.
Cara ini setara dengan menjalankan --verify enforced, tanpa membangkitkan ulang
jawaban (generator tidak deterministik, jadi justru lebih bersih: satu-satunya beda
dengan A0 adalah penggantian oleh layer).
"""
import csv
import json
from pathlib import Path

import common
from common import RESULTS, ENFORCED_FALLBACK, METRICS, load_dataset, ragas_score_rows

A0 = RESULTS / "A0_rerun"
OUT = RESULTS / "A4_enforced"


def main():
    OUT.mkdir(exist_ok=True)
    _, refs, _ = load_dataset()
    recs = {}
    for line in (A0 / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); recs[r["idx"]] = r
    a0 = json.loads((A0 / "ragas_results.json").read_text(encoding="utf-8"))
    assert len(a0) == 133 and len(recs) == 133
    rows, blocked = [], []
    for i in range(133):
        r = recs[i]
        v = r.get("verification")
        is_blocked = bool(v) and not v["is_valid"]
        ans = ENFORCED_FALLBACK if is_blocked else r["answer"]
        rows.append(dict(user_input=r["question"], retrieved_contexts=r["contexts"], response=ans, reference=refs[i]))
        if is_blocked:
            blocked.append(i)
    print(f"[*] diblokir: {len(blocked)}/133")
    sub = [rows[i] for i in blocked]
    sc = ragas_score_rows(sub, OUT / "ragas_blocked.jsonl", metric_names=["faithfulness", "answer_relevancy"],
                          label="A4_enforced", judge=common.build_judge(common.R1_JUDGE), judge_workers=5)
    final = []
    for i in range(133):
        rec = {**rows[i], **{m: a0[i].get(m) for m in METRICS}, "blocked": i in blocked}
        if i in blocked:
            s = sc[blocked.index(i)]
            rec["faithfulness"], rec["answer_relevancy"] = s.get("faithfulness"), s.get("answer_relevancy")
        final.append(rec)
    (OUT / "ragas_results.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "rag_run.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["question", "answer"] + [f"context_{j + 1}" for j in range(5)])
        for r in rows:
            w.writerow([r["user_input"], r["response"]] + r["retrieved_contexts"])

    # statistik pemblokiran
    f1 = [i for i in blocked if a0[i].get("faithfulness") == 1.0]
    fl = [i for i in blocked if a0[i].get("faithfulness") is not None and a0[i]["faithfulness"] < 1]
    fn = [i for i in blocked if a0[i].get("faithfulness") is None]
    lines = ["# A4 — verifikasi enforced", "",
             f"- Diblokir: {len(blocked)}/133 ({len(blocked) / 133:.1%})",
             f"- Di antaranya ber-faithfulness A0 = 1 (proxy 'sebenarnya benar' / tersokong konteks): {len(f1)}",
             f"- Ber-faithfulness A0 < 1: {len(fl)}; tanpa skor: {len(fn)}",
             f"- Nomor soal diblokir: {[i + 1 for i in blocked]}",
             f"- Judge: {common.R1_JUDGE} (sama dengan A0)",
             f"- Pesan fallback: \"{ENFORCED_FALLBACK}\""]
    (OUT / "A4_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
