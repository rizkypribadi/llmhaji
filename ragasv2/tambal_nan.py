"""
Menambal sel NaN pada hasil evaluasi RAGAS.
=============================================
Beberapa sel skor kosong (NaN) karena TimeoutError dari judge LLM saat run
penuh, bukan karena datanya bermasalah. Skrip ini menilai ulang HANYA sel-sel
tersebut, memakai jawaban dan konteks yang sudah tersimpan.

Bedanya dengan rescore_full.py: rescore_full menilai ulang seluruh 33 pertanyaan
x 5 metrik = 165 tugas (~2,8 jam). Skrip ini hanya menyentuh sel yang NaN —
biasanya kurang dari 10 tugas, selesai dalam hitungan menit. Sel yang sudah
punya skor TIDAK disentuh, sehingga angka yang sudah dilaporkan tetap stabil.

Pakai:
    cd ragasv2
    ..\\.venv\\Scripts\\python.exe -u tambal_nan.py
"""

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL

from eval_ragas import embeddings, JUDGE_MODEL

TARGET_FILE = "ragas_results_v2_deepseek.json"

# strictness=1 mengikuti konfigurasi run penuh: default (3) meminta 3 completion
# sekaligus per call, yang ditolak gateway sumopod dengan 400 Param Incorrect.
METRIC_FACTORY = {
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "context_entity_recall": ContextEntityRecall,
    "answer_relevancy": lambda: AnswerRelevancy(strictness=1),
    "faithfulness": Faithfulness,
}
METRICS = list(METRIC_FACTORY)


def kosong(nilai) -> bool:
    return nilai is None or (isinstance(nilai, float) and math.isnan(nilai))


def rata_rata(rows, metrik):
    nilai = [r[metrik] for r in rows if not kosong(r[metrik])]
    if not nilai:
        return float("nan"), 0
    return sum(nilai) / len(nilai), len(nilai)


def buat_judge():
    # max_tokens besar: deepseek-v4-pro adalah reasoning model, reasoning_tokens
    # dipotong dari anggaran yang sama sehingga nilai kecil membuat content kosong.
    return LangchainLLMWrapper(ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
        max_tokens=16384,
        max_retries=5,
    ))


def main():
    berkas = Path(TARGET_FILE)
    if not berkas.exists():
        print(f"[!] {TARGET_FILE} tidak ditemukan. Jalankan dari folder ragasv2/.")
        sys.exit(1)

    rows = json.loads(berkas.read_text(encoding="utf-8"))

    # Petakan sel NaN: metrik -> daftar indeks baris.
    kerja = {m: [i for i, r in enumerate(rows) if kosong(r.get(m))] for m in METRICS}
    kerja = {m: idx for m, idx in kerja.items() if idx}

    if not kerja:
        print("[*] Tidak ada sel NaN. Tidak ada yang perlu ditambal.")
        return

    total = sum(len(idx) for idx in kerja.values())
    print(f"[*] Judge: {JUDGE_MODEL}")
    print(f"[*] Sel NaN yang akan dinilai ulang: {total}")
    for m, idx in kerja.items():
        print(f"    |-- {m}: baris {idx}")
    print()

    # Cadangkan dulu. Kalau penambalan gagal di tengah, hasil asli tetap utuh.
    cadangan = berkas.with_suffix(".json.bak")
    shutil.copy2(berkas, cadangan)
    print(f"[*] Cadangan hasil asli -> {cadangan.name}\n")

    judge = buat_judge()
    # max_workers=1: sel-sel ini gagal justru karena timeout. Menilai satu per satu
    # menghilangkan rebutan koneksi ke gateway. Jumlahnya sedikit, jadi tetap cepat.
    run_config = RunConfig(max_workers=1, max_retries=15, max_wait=90, timeout=900)

    terisi, gagal = 0, []

    for metrik, indeks in kerja.items():
        print(f"[*] Menilai ulang {metrik} ({len(indeks)} baris)...")
        samples = [
            SingleTurnSample(
                user_input=rows[i]["user_input"],
                response=rows[i]["response"],
                retrieved_contexts=rows[i]["retrieved_contexts"],
                reference=rows[i]["reference"],
            )
            for i in indeks
        ]

        hasil = evaluate(
            EvaluationDataset(samples=samples),
            metrics=[METRIC_FACTORY[metrik]()],
            llm=judge,
            embeddings=embeddings,
            experiment_name=f"tambal_{metrik}",
            run_config=run_config,
        )

        skor = hasil.to_pandas()[metrik].tolist()
        for baris, nilai in zip(indeks, skor):
            if kosong(nilai):
                gagal.append((metrik, baris))
                print(f"    |-- baris {baris}: masih NaN")
            else:
                rows[baris][metrik] = float(nilai)
                terisi += 1
                print(f"    |-- baris {baris}: {float(nilai):.3f}")
        print()

    berkas.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 62)
    print(f"SELESAI — {terisi} dari {total} sel berhasil ditambal")
    if gagal:
        print(f"Masih NaN: {gagal}")
        print("Jalankan ulang skrip ini untuk mencoba lagi sel yang tersisa.")
    print("=" * 62)
    print(f"{'Metrik':<24}{'Rata-rata':>11}{'n':>5}")
    print("-" * 40)
    for m in METRICS:
        nilai, n = rata_rata(rows, m)
        tanda = "" if n == len(rows) else f"  (dari {len(rows)})"
        print(f"{m:<24}{nilai:>11.3f}{n:>5}{tanda}")
    print()
    print(f"Hasil diperbarui di {berkas.name}. Versi sebelum ditambal: {cadangan.name}")


if __name__ == "__main__":
    main()
