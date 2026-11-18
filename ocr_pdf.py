"""
OCR PDF pindaian -> Markdown untuk korpus Dataset/
==================================================
Dipakai untuk dua kasus di korpus ini:

1. PDF pindaian murni (nol lapisan teks) — mis. Buku Kemenhaj: PyPDFLoader
   mengekstrak 0 karakter, sehingga buku tidak pernah masuk indeks.
2. PDF yang punya lapisan teks tapi rusak — mis. "04 Panduan Praktis", yang
   menghasilkan "karni ielaskan"/"ko,uajiban" dan mencemari indeks.

Keluarannya Markdown (bukan .txt) karena load_documents() hanya memuat
PDF, MD, dan JSON.

Catatan lang: 'ind' saja, bukan 'ind+ara'. Uji 10 halaman pada Kemenhaj
menunjukkan penambahan 'ara' tidak menaikkan kualitas (0,230 vs 0,225) tetapi
menyisipkan karakter kontrol RTL yang mengotori embedding. Kutipan Arab gagal
di-OCR pada kedua setelan, jadi baris tersebut dibuang saat pembersihan.

Pakai:
    python ocr_pdf.py "Dataset/04 Panduan Praktis ....pdf"
    python ocr_pdf.py "<pdf>" --dpi 300 --lang ind

Prasyarat:
    winget install -e --id UB-Mannheim.TesseractOCR
    traineddata 'ind' (tessdata_best) di %USERPROFILE%\\tessdata
    pip install pytesseract pymupdf
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TESSDATA_PREFIX", str(Path.home() / "tessdata"))

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# Karakter Arab + tanda arah RTL: OCR latin menghasilkannya sebagai derau.
ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\u200e\u200f\u202a-\u202e]"
)


def is_garbage(line: str) -> bool:
    """Baris transliterasi Arab yang gagal, mis. 'KE Yee AA apa asi'.

    Ciri khasnya: banyak token sangat pendek dan acak. Prosa Indonesia yang
    sehat didominasi kata >3 huruf, jadi ambang 50% memisahkan keduanya bersih.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", line)
    if len(tokens) < 4:
        return False
    return sum(1 for t in tokens if len(t) <= 3) / len(tokens) >= 0.5


def clean(text: str) -> str:
    kept = []
    for line in text.splitlines():
        line = ARABIC_RE.sub("", line).rstrip()
        if not line.strip():
            kept.append("")
        elif not is_garbage(line):
            kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="path PDF sumber")
    ap.add_argument("--dpi", type=int, default=300, help="di bawah 300 akurasi anjlok")
    ap.add_argument("--lang", default="ind")
    ap.add_argument("--title", default=None, help="judul H1 di Markdown")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        sys.exit(f"[!] Tidak ditemukan: {pdf}")

    out_md = pdf.with_suffix(".md")
    out_raw = pdf.parent / "scanned_originals" / f"_ocr_raw_{pdf.stem}.txt"
    out_raw.parent.mkdir(exist_ok=True)

    doc = fitz.open(str(pdf))
    total = len(doc)
    print(f"[*] {pdf.name}: {total} halaman | DPI={args.dpi} | lang={args.lang}")
    print(f"[*] Output: {out_md.name}\n")

    parts = [f"# {args.title or pdf.stem}\n"]
    raw_parts = []
    dropped = 0
    t0 = time.time()

    for i in range(total):
        pix = doc[i].get_pixmap(dpi=args.dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        raw = pytesseract.image_to_string(img, lang=args.lang)
        raw_parts.append(f"\n--- HALAMAN {i + 1} ---\n{raw}")

        cleaned = clean(raw)
        dropped += len(raw.splitlines()) - len(cleaned.splitlines())
        if cleaned:
            parts.append(f"\n## Halaman {i + 1}\n\n{cleaned}\n")

        if (i + 1) % 25 == 0 or i + 1 == total:
            el = time.time() - t0
            eta = el / (i + 1) * (total - i - 1)
            print(f"    {i + 1:>4}/{total}  ({el / 60:.1f} mnt berlalu, ~{eta / 60:.1f} mnt lagi)")

    out_md.write_text("\n".join(parts), encoding="utf-8")
    out_raw.write_text("\n".join(raw_parts), encoding="utf-8")

    chars = len(out_md.read_text(encoding="utf-8"))
    print(f"\n[OK] Selesai dalam {(time.time() - t0) / 60:.1f} menit")
    print(f"     {chars:,} karakter -> perkiraan ~{chars // 1000} chunk @1000 char")
    print(f"     {dropped:,} baris derau dibuang | mentah: {out_raw.name}")


if __name__ == "__main__":
    main()
