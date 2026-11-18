"""Ringkasan statistik hasil RAGAS per varian.

Untuk tiap metrik: n, mean, SD per pertanyaan, median, bootstrap 95% CI
(B = 10.000, seed 42, resampling per pertanyaan). Bila --baseline diberikan:
selisih berpasangan (hanya soal yang punya skor di KEDUA run) dengan
paired-bootstrap 95% CI + p-value dua sisi, dan Wilcoxon signed-rank.

NaN dikeluarkan hanya dari metrik itu; jumlahnya dilaporkan (n_nan).

Pakai:
    python revisi_r1/summarize.py revisi_r1/results/A1_unconstrained/ragas_results.json \
        --baseline revisi_r1/results/A0_rerun/ragas_results.json --out A1_vs_A0.csv
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["context_precision", "context_recall", "context_entity_recall",
           "answer_relevancy", "faithfulness"]
B, SEED = 10_000, 42


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    df = pd.DataFrame(rows)
    for m in METRICS:
        if m not in df:
            df[m] = np.nan
        df[m] = pd.to_numeric(df[m], errors="coerce")
    # Konvensi run final: jawaban kosong -> faithfulness tidak terdefinisi (NaN), answer_relevancy = 0.
    # Judge kadang tetap memberi angka untuk jawaban kosong (mis. 0.75), jadi diseragamkan di sini.
    if "response" in df:
        df.loc[df["response"].fillna("").str.strip() == "", "faithfulness"] = np.nan
    return df


def boot_ci(v, rng):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (np.nan, np.nan)
    means = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def describe(df, metrics=METRICS):
    out = []
    for m in metrics:
        rng = np.random.default_rng(SEED)
        v = df[m].dropna().to_numpy()
        lo, hi = boot_ci(v, rng)
        out.append(dict(metric=m, n=len(v), n_nan=int(df[m].isna().sum()), mean=v.mean() if len(v) else np.nan,
                        sd=v.std(ddof=1) if len(v) > 1 else np.nan, median=np.median(v) if len(v) else np.nan,
                        ci_low=lo, ci_high=hi))
    return pd.DataFrame(out)


def paired(df, base, metrics=METRICS):
    assert len(df) == len(base), "jumlah soal berbeda"
    out = []
    for m in metrics:
        rng = np.random.default_rng(SEED)
        mask = df[m].notna() & base[m].notna()
        d = (df[m][mask] - base[m][mask]).to_numpy()
        if len(d) == 0:
            out.append(dict(metric=m, n_pair=0)); continue
        bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
        # p dua sisi: proporsi bootstrap di sisi berlawanan dari 0, digandakan
        p_boot = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        try:
            p_w = wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p_w = np.nan
        out.append(dict(metric=m, n_pair=len(d), mean_variant=df[m][mask].mean(), mean_base=base[m][mask].mean(),
                        delta=d.mean(), delta_ci_low=np.percentile(bs, 2.5), delta_ci_high=np.percentile(bs, 97.5),
                        p_bootstrap=p_boot, p_wilcoxon=p_w, n_up=int((d > 0).sum()), n_down=int((d < 0).sum())))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = load(a.results)
    present = [m for m in METRICS if df[m].notna().any()]
    desc = describe(df, present)
    pd.set_option("display.width", 200)
    print(f"== {a.results}  (N soal = {len(df)})")
    print(desc.round(4).to_string(index=False))
    res = desc
    if a.baseline:
        base = load(a.baseline)
        pr = paired(df, base, present)
        print(f"\n== vs baseline {a.baseline}")
        print(pr.round(4).to_string(index=False))
        res = desc.merge(pr, on="metric", how="left")
    if a.out:
        res.round(6).to_csv(a.out, index=False)
        print(f"\n[ok] {a.out}")


if __name__ == "__main__":
    main()
