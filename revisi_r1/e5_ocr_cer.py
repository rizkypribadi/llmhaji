"""E5 — CER/WER Tesseract 'ind' vs 'ind+ara' tanpa transkripsi manual.

Ground truth = lapisan teks digital PDF yang bersih. Di korpus hanya
'05 Manasik Haji dan Umrah - Terjemah Manhajus Salikin.pdf' (InDesign, born-digital)
yang memenuhi syarat; PDF lain berupa pindaian dengan lapisan OCR kotor
(diperiksa: font Helvetica/GlyphLess + gambar menutup 100% halaman).

Langkah: sampel acak N halaman (seed 42, hanya halaman dengan >= 200 karakter teks),
rasterisasi pada DPI tertentu, OCR dengan ocr_pdf.py setelan yang sama (pytesseract,
default PSM), lalu CER/WER (jiwer) setelah normalisasi: NFKC, huruf kecil, tanda baca
dibuang, spasi diciutkan. Dilaporkan dua cakupan:
  - 'latin'  : karakter Arab dibuang dari GT dan hipotesis (sesuai pipeline yang membuang Arab)
  - 'penuh'  : teks apa adanya (Arab ikut dinilai)
Juga dihitung karakter kontrol arah (U+200E/F, U+202A-E, U+2066-9) di keluaran mentah.

Juga menyiapkan 5 halaman Kemenhaj (pindaian asli) + template transkripsi manual.
"""
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TESSDATA_PREFIX", str(Path.home() / "tessdata"))

import jiwer
import pymupdf
import pytesseract
from PIL import Image

from ocr_pdf import TESSERACT_EXE, clean  # setelan & pembersih yang sama dengan pipeline
if Path(TESSERACT_EXE).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

OUT = ROOT / "revisi_r1/results/E5_ocr"
GT_PDF = ROOT / "Dataset/05 Manasik Haji dan Umrah - Terjemah Manhajus Salikin.pdf"
SCAN_PDF = ROOT / "Dataset/scanned_originals/06 Buku Tuntunan Manasik Haji Kemenhaj.pdf"
SEED, N_PAGES = 42, 25
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
BIDI = re.compile(r"[‎‏‪-‮⁦-⁩]")


def norm(s, latin_only):
    s = unicodedata.normalize("NFKC", s)
    s = BIDI.sub("", s)
    if latin_only:
        s = ARABIC.sub(" ", s)
    s = s.lower()
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ocr(page, dpi, lang):
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang=lang)


def score(gt, hyp, latin):
    g, h = norm(gt, latin), norm(hyp, latin)
    if not g:
        return None
    return dict(cer=jiwer.cer(g, h), wer=jiwer.wer(g, h), gt_chars=len(g))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(GT_PDF))
    eligible = [i for i in range(len(doc)) if len(doc[i].get_text().strip()) >= 200]
    pages = sorted(random.Random(SEED).sample(eligible, min(N_PAGES, len(eligible))))
    print(f"[*] {GT_PDF.name}: {len(doc)} hal, layak {len(eligible)}, sampel {len(pages)}: {[p + 1 for p in pages]}")
    configs = [(300, "ind"), (300, "ind+ara"), (200, "ind"), (150, "ind")]
    per = []
    for p in pages:
        gt = doc[p].get_text()
        row = {"page": p + 1}
        for dpi, lang in configs:
            raw = ocr(doc[p], dpi, lang)
            key = f"{lang}@{dpi}"
            row[key] = {"bidi_chars": len(BIDI.findall(raw)), "arabic_chars": len(ARABIC.findall(raw)),
                        "latin_raw": score(gt, raw, True), "penuh_raw": score(gt, raw, False),
                        "latin_clean": score(gt, clean(raw), True)}
        per.append(row)
        print(f"  hal {p + 1}: " + " | ".join(f"{k} CER={row[k]['latin_raw']['cer']:.3f}" for k in row if k != "page"),
              flush=True)
    (OUT / "per_page.json").write_text(json.dumps(per, indent=2), encoding="utf-8")

    def agg(key, sub, field):
        v = [r[key][sub][field] for r in per if r[key][sub]]
        w = [r[key][sub]["gt_chars"] for r in per if r[key][sub]]
        import numpy as np
        return float(np.mean(v)), float(np.average(v, weights=w)), float(np.median(v))

    lines = ["# E5 — CER/WER OCR", "", f"Sumber GT: {GT_PDF.name} (lapisan teks digital). N = {len(per)} halaman, seed {SEED}.",
             f"Halaman: {[r['page'] for r in per]}", "",
             "| Setelan | Cakupan | CER mean | CER (bobot char) | CER median | WER mean | WER median | karakter kontrol RTL (total) |",
             "|---|---|---|---|---|---|---|---|"]
    for dpi, lang in configs:
        k = f"{lang}@{dpi}"
        bidi = sum(r[k]["bidi_chars"] for r in per)
        for sub in ("latin_raw", "latin_clean", "penuh_raw"):
            c = agg(k, sub, "cer"); w = agg(k, sub, "wer")
            lines.append(f"| {k} | {sub} | {c[0]:.4f} | {c[1]:.4f} | {c[2]:.4f} | {w[0]:.4f} | {w[2]:.4f} | {bidi} |")
    lines += ["", "Catatan: halaman dirender dari PDF digital (bersih, tanpa derau pindai), sehingga angka ini "
              "adalah batas bawah (optimistis) dibanding pindaian nyata seperti Kemenhaj."]
    (OUT / "E5_ocr.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # --- sampel manual Kemenhaj ---
    man = ROOT / "revisi_r1/ocr_manual"
    man.mkdir(parents=True, exist_ok=True)
    sd = pymupdf.open(str(SCAN_PDF))
    cand = list(range(30, len(sd) - 10))  # lewati sampul/daftar isi
    picks = sorted(random.Random(SEED).sample(cand, 5))
    tmpl = ["# Transkripsi manual — 5 halaman Kemenhaj (pindaian asli)", "",
            "Ketik ulang isi setiap halaman PERSIS seperti tercetak (abaikan nomor halaman & header).",
            "Teks Arab boleh ditulis [ARAB] saja. Simpan file ini; CER/WER dihitung otomatis.", ""]
    for p in picks:
        pix = sd[p].get_pixmap(dpi=300)
        pix.save(str(man / f"kemenhaj_hal{p + 1}.png"))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        for lang in ("ind", "ind+ara"):
            (man / f"kemenhaj_hal{p + 1}_ocr_{lang.replace('+', '_')}.txt").write_text(
                pytesseract.image_to_string(img, lang=lang), encoding="utf-8")
        tmpl += [f"## Halaman {p + 1}  (gambar: kemenhaj_hal{p + 1}.png)", "", "```", "", "```", ""]
    (man / "TRANSKRIPSI_TEMPLATE.md").write_text("\n".join(tmpl), encoding="utf-8")
    print(f"[ok] sampel manual: halaman {[p + 1 for p in picks]} -> {man}")


if __name__ == "__main__":
    main()
