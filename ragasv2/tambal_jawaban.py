"""
Menambal jawaban kosong pada CSV hasil pipeline RAG
====================================================
Sesekali LLM penjawab mengembalikan string kosong karena gangguan API sesaat.
Kalau dibiarkan, metrik faithfulness dan answer_relevancy untuk baris itu
menjadi NaN atau nol palsu.

Skrip ini membangkitkan ulang HANYA baris yang jawabannya kosong, memakai
konteks yang sudah tersimpan di CSV — retrieval dan reranking tidak diulang,
sehingga konteksnya persis sama dan skornya tetap sebanding.

Pakai:
    cd ragasv2
    ..\\.venv\\Scripts\\python.exe -u tambal_jawaban.py                  # CSV terbaru
    ..\\.venv\\Scripts\\python.exe -u tambal_jawaban.py rag_run_xxx.csv
"""

import csv
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from chat_engine_v2 import get_chat_client, generate_response
from eval_ragas import GAYA_EVALUASI

csv.field_size_limit(10 ** 7)


def csv_terbaru() -> Path:
    berkas = sorted(Path(".").glob("rag_run_*.csv"), key=lambda p: p.stat().st_mtime)
    if not berkas:
        print("[!] Tidak ada rag_run_*.csv di folder ini.")
        sys.exit(1)
    return berkas[-1]


def main():
    sumber = Path(sys.argv[1]) if len(sys.argv) > 1 else csv_terbaru()
    print(f"[*] Berkas : {sumber.name}")
    print(f"[*] Gaya   : {GAYA_EVALUASI}")

    with open(sumber, encoding="utf-8") as f:
        pembaca = csv.DictReader(f)
        kolom = pembaca.fieldnames
        baris = list(pembaca)

    kolom_ctx = [k for k in kolom if k.startswith("context_")]
    kosong = [i for i, b in enumerate(baris) if not (b.get("answer") or "").strip()]

    if not kosong:
        print("[*] Tidak ada jawaban kosong. Tidak ada yang perlu ditambal.")
        return

    print(f"[*] Jawaban kosong pada baris: {[i + 1 for i in kosong]}\n")

    cadangan = sumber.with_suffix(".csv.bak")
    shutil.copy2(sumber, cadangan)
    print(f"[*] Cadangan -> {cadangan.name}\n")

    client = get_chat_client()
    gagal = []

    for i in kosong:
        b = baris[i]
        konteks = "\n\n".join(b[k] for k in kolom_ctx if (b.get(k) or "").strip())
        print(f"[*] Baris {i + 1}: {b['question'][:70]}")

        jawab = "".join(generate_response(
            client, b["question"], konteks, stream=True, gaya=GAYA_EVALUASI
        )).strip()

        if jawab:
            baris[i]["answer"] = jawab
            print(f"    |-- {len(jawab)} karakter: {jawab[:110]}\n")
        else:
            gagal.append(i + 1)
            print("    |-- masih kosong\n")

    with open(sumber, "w", newline="", encoding="utf-8") as f:
        penulis = csv.DictWriter(f, fieldnames=kolom)
        penulis.writeheader()
        penulis.writerows(baris)

    sisa = [i + 1 for i, b in enumerate(baris) if not (b.get("answer") or "").strip()]
    print("=" * 58)
    print(f"SELESAI — {len(kosong) - len(gagal)} dari {len(kosong)} jawaban ditambal")
    if sisa:
        print(f"Masih kosong: {sisa}. Jalankan ulang skrip ini untuk mencoba lagi.")
    print("=" * 58)


if __name__ == "__main__":
    main()
