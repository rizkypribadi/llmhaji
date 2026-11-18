"""Analisis E1 (judge berulang) dan E2 (judge independen).

E1: untuk tiap metrik -> rerata tiap run, rerata dari k run, SD antar-run (σ_run, ddof=1),
    bootstrap 95% CI atas rerata per-pertanyaan (rata-rata k run per soal; B=10.000, seed 42),
    plus SD per-pertanyaan rata-rata (variasi judge pada soal yang sama).
E2: judge independen vs deepseek (run final): rerata + CI, selisih berpasangan (+CI, Wilcoxon),
    Spearman per pertanyaan, Cohen's κ untuk faithfulness biner (=1 vs <1), dan daftar 10 soal
    dengan selisih faithfulness terbesar.

    python revisi_r1/analyze_judges.py e1
    python revisi_r1/analyze_judges.py e2
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import load, describe, paired, boot_ci, METRICS, SEED  # noqa: E402

R = Path(__file__).resolve().parent / "results"
FINAL = Path(__file__).resolve().parent.parent / "ragasv2" / "ragas_results_v4_deepseek.json"


def e1(name="E1_flashlite31"):
    runs = sorted((R / name).glob("run*/ragas_results.json"))
    dfs = [load(p) for p in runs]
    k = len(dfs)
    final = load(FINAL)
    lines = [f"# E1 — judge berulang ({name}), k = {k} (jawaban run final yang sama)", "",
             "| Metrik | " + " | ".join(f"run{i + 1}" for i in range(k)) +
             " | rerata k run | σ_run (SD antar-run) | 95% CI (bootstrap, rerata per soal) | rerata SD per soal | final Agustus |",
             "|---|" + "---|" * (k + 5)]
    out = []
    for m in METRICS:
        means = [d[m].mean() for d in dfs]
        mat = np.vstack([d[m].to_numpy(float) for d in dfs])          # k x 133
        q_mean = np.nanmean(mat, axis=0)
        q_mean = q_mean[~np.isnan(q_mean)]
        lo, hi = boot_ci(q_mean, np.random.default_rng(SEED))
        sd_q = np.nanmean(np.nanstd(mat, axis=0, ddof=1)) if k > 1 else np.nan
        sd_run = np.std(means, ddof=1) if k > 1 else np.nan
        ns = [int(d[m].notna().sum()) for d in dfs]
        out.append(dict(metric=m, run_means=means, n_per_run=ns, mean_of_runs=float(np.mean(means)),
                        sd_across_runs=float(sd_run), ci_low=lo, ci_high=hi, mean_per_question_sd=float(sd_q),
                        final_aug=float(final[m].mean())))
        lines.append(f"| {m} | " + " | ".join(f"{x:.4f} (n={n})" for x, n in zip(means, ns)) +
                     f" | {np.mean(means):.4f} | {sd_run:.4f} | [{lo:.4f}, {hi:.4f}] | {sd_q:.4f} | {final[m].mean():.4f} |")
    (R / name / "E1_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (R / name / "E1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def e2(name="E1_flashlite31"):  # E2 = run1 dari E1 (judge & jawaban identik)
    ind = load(R / name / "run1" / "ragas_results.json")
    ds = load(FINAL)
    d_ind, d_ds = describe(ind), describe(ds)
    pr = paired(ind, ds)
    lines = [f"# E2 — judge independen ({name}) vs deepseek-v4-pro (run final)", "",
             "| Metrik | n | independen mean [95% CI] | deepseek mean [95% CI] | Δ (ind − ds) [95% CI] | p Wilcoxon | Spearman ρ (p) |",
             "|---|---|---|---|---|---|---|"]
    for m in METRICS:
        a, b, p = d_ind.set_index("metric").loc[m], d_ds.set_index("metric").loc[m], pr.set_index("metric").loc[m]
        mask = ind[m].notna() & ds[m].notna()
        rho, pv = spearmanr(ind[m][mask], ds[m][mask])
        lines.append(f"| {m} | {int(p.n_pair)} | {a['mean']:.4f} [{a.ci_low:.4f}, {a.ci_high:.4f}] | "
                     f"{b['mean']:.4f} [{b.ci_low:.4f}, {b.ci_high:.4f}] | {p.delta:+.4f} [{p.delta_ci_low:+.4f}, "
                     f"{p.delta_ci_high:+.4f}] | {p.p_wilcoxon:.4g} | {rho:.3f} ({pv:.3g}) |")
    mask = ind["faithfulness"].notna() & ds["faithfulness"].notna()
    bi, bd = (ind["faithfulness"][mask] >= 1).astype(int), (ds["faithfulness"][mask] >= 1).astype(int)
    kappa = cohen_kappa_score(bi, bd)
    agree = (bi == bd).mean()
    ct = pd.crosstab(bd.rename("deepseek=1"), bi.rename("independen=1"))
    lines += ["", f"**Faithfulness biner (=1 vs <1):** Cohen's κ = {kappa:.3f}; kesepakatan mentah {agree:.1%} (n = {mask.sum()})",
              "", "```", ct.to_string(), "```", "", "## 10 soal dengan selisih faithfulness terbesar", "",
              "| No | deepseek | independen | Δ | pertanyaan |", "|---|---|---|---|---|"]
    diff = (ind["faithfulness"] - ds["faithfulness"]).abs()
    for i in diff.dropna().sort_values(ascending=False).index[:10]:
        lines.append(f"| {i + 1} | {ds.faithfulness[i]:.3f} | {ind.faithfulness[i]:.3f} | "
                     f"{ind.faithfulness[i] - ds.faithfulness[i]:+.3f} | {ds.user_input[i][:90]} |")
    (R / name / "E2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    {"e1": e1, "e2": e2}[sys.argv[1]](*sys.argv[2:])
