"""E3-A3 — korpus dengan OCR 'ind+ara' -> koleksi baru mutawwif_v4_indara.

Satu-satunya beda dengan mutawwif_v4: dua kitab pindaian yang PDF sumbernya ada
(04 Panduan Praktis, 152 hlm; 06 Kemenhaj, 510 hlm) di-OCR ulang dengan
lang='ind+ara' (DPI 300, clean() yang sama dari ocr_pdf.py, judul H1 yang sama).
02 Nawawi tetap versi 'ind' (PDF sumber tidak tersedia). Kitab lain dan wawancara
identik. Chunking (recursive 1000/200) dan embedding (bge-m3) sama dengan reindex_v4.py.

Tidak menyentuh Dataset/ maupun koleksi lama. Korpus hasil disimpan di
results/A3_indara/corpus/ (berhak cipta; di-.gitignore).

    python revisi_r1/a3_build_indara.py ocr      # resumable per halaman
    python revisi_r1/a3_build_indara.py corpus
    python revisi_r1/a3_build_indara.py index    # resumable per batch (id deterministik)
"""
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = ROOT / "revisi_r1" / "results" / "A3_indara"
CACHE = OUT / "ocr_cache"
CORPUS = OUT / "corpus"
DATASET = ROOT / "Dataset"
SCANS = DATASET / "scanned_originals"
COLLECTION = "mutawwif_v4_indara"
PROTECTED = {"mutawwif", "mutawwif_v2", "mutawwif_v3", "mutawwif_v4"}
LANG, DPI = "ind+ara", 300
BOOKS = {  # kunci -> nama file (stem) di Dataset/ dan scanned_originals/
    "04": "04 Panduan Praktis Haji, Umrah, dan Ziarah Berdasarkan Al-Qur_an dan Sunnah",
    "06": "06 Buku Tuntunan Manasik Haji Kemenhaj",
}


def _ocr_page(args):
    pdf, i, dst = args
    import fitz
    import pytesseract
    from PIL import Image
    import ocr_pdf  # noqa: F401  (set TESSDATA_PREFIX & tesseract_cmd)
    doc = fitz.open(pdf)
    pix = doc[i].get_pixmap(dpi=DPI)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    raw = pytesseract.image_to_string(img, lang=LANG)
    tmp = Path(dst).with_suffix(".tmp")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(dst)
    return i


def ocr(workers=8):
    import fitz
    jobs = []
    for k, stem in BOOKS.items():
        pdf = SCANS / f"{stem}.pdf"
        n = len(fitz.open(str(pdf)))
        d = CACHE / k
        d.mkdir(parents=True, exist_ok=True)
        jobs += [(str(pdf), i, str(d / f"p{i + 1:04d}.txt")) for i in range(n)
                 if not (d / f"p{i + 1:04d}.txt").exists()]
        print(f"[*] {k}: {n} halaman")
    print(f"[*] {len(jobs)} halaman perlu di-OCR (lang={LANG}, dpi={DPI}, {workers} proses)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_ocr_page, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            f.result()
            if n % 25 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"    {n}/{len(jobs)}  {el / 60:.1f} mnt, ~{el / n * (len(jobs) - n) / 60:.1f} mnt lagi", flush=True)


def corpus():
    from ocr_pdf import clean
    if CORPUS.exists():
        shutil.rmtree(CORPUS)
    CORPUS.mkdir(parents=True)
    for p in DATASET.iterdir():  # sama dengan glob non-rekursif load_documents
        if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".json"}:
            shutil.copy2(p, CORPUS / p.name)
    for k, stem in BOOKS.items():
        old = (DATASET / f"{stem}.md").read_text(encoding="utf-8")
        parts = [old.splitlines()[0] + "\n"]  # judul H1 yang sama persis
        pages = sorted((CACHE / k).glob("p*.txt"))
        for p in pages:
            c = clean(p.read_text(encoding="utf-8"))
            if c:
                parts.append(f"\n## Halaman {int(p.stem[1:])}\n\n{c}\n")
        new = "\n".join(parts)
        (CORPUS / f"{stem}.md").write_text(new, encoding="utf-8")
        print(f"[*] {k}: {len(pages)} hlm | ind {len(old):,} char -> ind+ara {len(new):,} char")
    print("[*] korpus:", sorted(x.name for x in CORPUS.iterdir()))


def index(batch=10):
    assert COLLECTION not in PROTECTED
    from data_loader import load_documents, split_documents
    from vector_manager import get_another_vector_store
    from qdrant_client import QdrantClient
    from config import QDRANT_URL, QDRANT_API_KEY
    chunks = split_documents(load_documents(str(CORPUS)), strategy="recursive")
    print(f"[*] {len(chunks)} chunk (v4: 4.470)", flush=True)
    vs = get_another_vector_store(force_reindex=False, collection_name=COLLECTION)
    cl = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    have = cl.count(COLLECTION, exact=True).count
    start = (have // batch) * batch  # upsert ulang batch terakhir yang mungkin parsial
    print(f"[*] koleksi {COLLECTION}: {have} titik; mulai dari chunk {start}", flush=True)
    ns = uuid.UUID("6f1c1d3e-4a1b-4c55-9a39-a3a3a3a3a3a3")
    t0 = time.time()
    for i in range(start, len(chunks), batch):
        b = chunks[i:i + batch]
        vs.add_documents(b, ids=[str(uuid.uuid5(ns, str(i + j))) for j in range(len(b))])
        if (i // batch) % 20 == 0:
            el = time.time() - t0
            done = i + len(b) - start
            print(f"    {i + len(b)}/{len(chunks)}  {el / 60:.1f} mnt, ~{el / done * (len(chunks) - i - len(b)) / 60:.1f} mnt lagi", flush=True)
    n = cl.count(COLLECTION, exact=True).count
    print(f"[OK] {COLLECTION}: {n} titik (harus = {len(chunks)})")
    assert n == len(chunks)


if __name__ == "__main__":
    {"ocr": ocr, "corpus": corpus, "index": index}[sys.argv[1]]()
