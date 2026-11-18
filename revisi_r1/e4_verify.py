"""E4 — Statistik verification layer pada 133 pasangan jawaban+konteks run final.

Memakai modul ASLI hallucination_checker.HallucinationChecker dengan model yang
dipakai produksi (app.py: nli_model=NLI_MODEL, embedding_model=EMBEDDING_MODEL).
Tidak ada logika yang diimplementasi ulang; sweep ambang memanggil _make_decision
asli setelah mengganti atribut kelas ENTAILMENT_THRESHOLD.

Acuan "halusinasi" (positif):
  (i)  RAGAS faithfulness < 1 (ragas_results_v4_deepseek.json)
  (ii) label ahli 'Bertentangan dengan GT' dsb. dari Anotasi_Perbandingan_133.xlsx (bila sudah terisi)

Diagnostik tambahan: panjang token pasangan (premis, hipotesis) pada tokenizer NLI
untuk menunjukkan berapa yang terpotong di 512 token.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "revisi_r1/results/E4_verification"
OUT.mkdir(parents=True, exist_ok=True)


def main(src=ROOT / "ragasv2/ragas_results_v4_deepseek.json", tag="final"):
    from config import NLI_MODEL, EMBEDDING_MODEL
    from hallucination_checker import HallucinationChecker
    rows = json.loads(Path(src).read_text(encoding="utf-8"))
    hc = HallucinationChecker(nli_model_name=NLI_MODEL, embedding_model_name=EMBEDDING_MODEL)
    tok = hc.nli_model.tokenizer
    maxlen = getattr(hc.nli_model, "max_length", None) or tok.model_max_length

    recs = []
    for i, r in enumerate(rows):
        ans = (r.get("response") or "").strip()
        ctx = "\n\n".join(r["retrieved_contexts"]).strip()
        rec = {"no": i + 1, "faithfulness": r.get("faithfulness"), "empty_answer": not ans}
        n_prem = len(tok(ctx, add_special_tokens=False)["input_ids"])
        n_hyp = len(tok(ans, add_special_tokens=False)["input_ids"]) if ans else 0
        rec.update(tokens_premise=n_prem, tokens_hypothesis=n_hyp, truncated=(n_prem + n_hyp + 3) > maxlen)
        if ans:
            v = hc.verify_answer(answer=ans, context=ctx, query=r["user_input"])
            rec.update({k: (bool(x) if isinstance(x, (bool, np.bool_)) else x) for k, x in v.items()})
            rec["safe_fallback_rule"] = bool(hc._is_safe_fallback(ans))
        recs.append(rec)
    (OUT / f"verdicts_{tag}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=2, default=float),
                                              encoding="utf-8")

    ev = [r for r in recs if not r["empty_answer"]]
    n = len(ev)
    flagged = [r for r in ev if not r["is_valid"]]
    lines = [f"# E4 — verification layer pada run {tag}", "",
             f"Jawaban dinilai: {n} (jawaban kosong dikeluarkan: {len(recs) - n}); max_length NLI = {maxlen}", "",
             f"**Flag rate:** {len(flagged)}/{n} = {len(flagged) / n:.1%}", "",
             f"Pasangan NLI yang terpotong (> {maxlen} token): {sum(r['truncated'] for r in ev)}/{n}; "
             f"median token premis {int(np.median([r['tokens_premise'] for r in ev]))}, "
             f"hipotesis {int(np.median([r['tokens_hypothesis'] for r in ev]))}", ""]

    def confusion(pos_fn, label):
        sc = [r for r in ev if pos_fn(r) is not None]
        tp = sum(1 for r in sc if pos_fn(r) and not r["is_valid"])
        fn = sum(1 for r in sc if pos_fn(r) and r["is_valid"])
        fp = sum(1 for r in sc if not pos_fn(r) and not r["is_valid"])
        tn = sum(1 for r in sc if not pos_fn(r) and r["is_valid"])
        P, N = tp + fn, fp + tn
        lines.extend([f"## Acuan: {label} (n = {len(sc)}; positif = {P}, negatif = {N})", "",
                      "| | flagged | lolos |", "|---|---|---|",
                      f"| positif (halusinasi) | TP={tp} | FN={fn} |", f"| negatif | FP={fp} | TN={tn} |", "",
                      f"- Recall = {tp}/{P} = {tp / P:.3f}" if P else "- Recall: tidak terdefinisi (tidak ada positif)",
                      f"- FPR = {fp}/{N} = {fp / N:.3f}" if N else "- FPR: tidak terdefinisi",
                      f"- Precision = {tp}/{tp + fp} = {tp / (tp + fp):.3f}" if tp + fp else "- Precision: -", ""])

    confusion(lambda r: None if r["faithfulness"] is None else r["faithfulness"] < 1, "RAGAS faithfulness < 1")

    # kontribusi tiap pemeriksaan
    lines += ["## Kontribusi pemeriksaan", "",
              f"- Lolos via aturan safe-fallback/greeting (substring): {sum(1 for r in ev if r['safe_fallback_rule'])}",
              f"- NLI ≥ 0,65: {sum(1 for r in ev if r['entailment_score'] >= 0.65)}; 0,50–0,65: "
              f"{sum(1 for r in ev if 0.5 <= r['entailment_score'] < 0.65)}; < 0,50: {sum(1 for r in ev if r['entailment_score'] < 0.5)}",
              f"- Overlap ≥ 0,40: {sum(1 for r in ev if r['overlap_score'] >= 0.40)} (median {np.median([r['overlap_score'] for r in ev]):.3f})",
              f"- Citation check lolos: {sum(1 for r in ev if r['has_citations'])}",
              f"- Keyword check lolos: {sum(1 for r in ev if r['has_fiqh_terms'])}", "",
              "Alasan keputusan (validation_details, dikelompokkan):", ""]
    from collections import Counter
    c = Counter(r["validation_details"].split("(")[0].strip() for r in ev)
    lines += [f"- {k}: {v}" for k, v in c.most_common()] + [""]

    # sweep ambang NLI memakai _make_decision asli
    lines += ["## Threshold sweep τ_NLI (acuan RAGAS faithfulness < 1)", "",
              "| τ_NLI | flag rate | recall | FPR |", "|---|---|---|---|"]
    orig = type(hc).ENTAILMENT_THRESHOLD
    sweep = []
    sc = [r for r in ev if r["faithfulness"] is not None]
    for t in np.round(np.arange(0.3, 0.91, 0.05), 2):
        type(hc).ENTAILMENT_THRESHOLD = float(t)
        dec = [hc._make_decision(r["entailment_score"], r["overlap_score"], r["has_citations"],
                                 r["has_fiqh_terms"], r["safe_fallback_rule"])[0] for r in sc]
        pos = [r["faithfulness"] < 1 for r in sc]
        tp = sum(p and not d for p, d in zip(pos, dec)); P = sum(pos)
        fp = sum((not p) and not d for p, d in zip(pos, dec)); N = len(pos) - P
        fr = sum(not d for d in dec) / len(dec)
        sweep.append(dict(tau=float(t), flag_rate=fr, recall=tp / P if P else None, fpr=fp / N if N else None))
        lines.append(f"| {t:.2f} | {fr:.3f} | {tp / P if P else float('nan'):.3f} | {fp / N if N else float('nan'):.3f} |")
    type(hc).ENTAILMENT_THRESHOLD = orig
    lines.append("\nCatatan: ambang moderat 0,50 di _make_decision tertanam di kode (bukan atribut), sehingga "
                 "untuk τ < 0,50 cabang 'moderat' tidak pernah aktif.")
    (OUT / f"sweep_{tag}.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    (OUT / f"E4_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
