"""E7.2 — Apakah entitas yang 'tidak ditemukan' ada di korpus?

Mencari setiap entitas secara full-text di (a) payload seluruh chunk Qdrant
mutawwif_v4 dan (b) teks mentah Dataset/ (MD langsung, PDF via PyPDFLoader
seperti data_loader). Pencocokan insensitif huruf dan varian ejaan:
huruf kecil, aksen dibuang, apostrof dibuang, dz->z, dh->d, th->t, ch->h,
sh->s, kh->h, e->a, huruf ganda diciutkan, lalu SEMUA spasi/tanda baca dibuang
(sehingga 'Al-Juhfah' == 'al juhfah' == 'aljuhfah').
"""
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ENTITAS = {  # soal -> label -> varian yang dicari
    "20": {"Jumrah Aqabah": ["jumrah aqabah", "jamrah aqabah", "jumratul aqabah", "jamratul aqabah", "jumrotul aqobah"]},
    "21": {"Umar bin Khattab": ["umar bin khattab", "umar ibn khattab", "umar bin al khattab", "umar ibnul khattab"],
           "Masjidil Haram": ["masjidil haram", "masjid al haram", "masjidilharam", "masjid haram"]},
    "24,25,30": {"Muzakarah Perhajian": ["muzakarah perhajian", "mudzakarah perhajian", "muzakarah haji", "mudzakarah haji"],
                 "murur (konteks)": ["murur"], "tanazul (konteks)": ["tanazul"]},
    "31": {"Daker Makkah": ["daker makkah", "daker mekkah", "daerah kerja makkah", "daerah kerja mekkah"]},
    "34": {"Al Juhfah": ["juhfah"]},
    "39": {"Imam Al Bukhari": ["imam bukhari", "imam al bukhari", "al bukhari"], "Bukhari (umum)": ["bukhari"]},
    "66": {"Dam Ihshar": ["dam ihshar", "dam ichshor", "dam ihsar", "dam ikhshar"], "ihshar (umum)": ["ihshar", "ihsar", "ichshor", "muhshar", "ikhshar"]},
    "83": {"Ka'ab bin Ujrah": ["kaab bin ujrah", "kab bin ujrah", "kaab ibn ujrah", "ka ab bin ujrah"], "Ujrah (umum)": ["ujrah"]},
    "85": {"Ramadan": ["ramadan", "ramadhan"]},
    "90": {"Duba'ah": ["dubaah", "dhubaah", "dhuba ah", "dubaah binti zubair"]},
    "128": {"38.0 °C": ["38 0 c", "38 c", "38 derajat", "38 0 derajat", "38°c", "38 0°c"],
            "heat stroke": ["heat stroke", "heatstroke", "sengatan panas", "serangan panas"]},
    "131": {"Imam Qurtubi": ["imam qurtubi", "imam qurthubi", "al qurthubi", "al qurtubi"], "Qurtubi (umum)": ["qurtubi", "qurthubi"]},
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("°", " derajatsym ")
    s = re.sub(r"['’‘`ʼʻ]", "", s)
    for a, b in (("dz", "z"), ("dh", "d"), ("th", "t"), ("ch", "h"), ("sh", "s"), ("kh", "h"), ("e", "a"), ("o", "u")):
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)  # ciutkan huruf ganda SETELAH spasi dibuang (urutan konsisten)


def norm_needle(v):
    return norm(v.replace("°c", " derajatsym c"))


def main():
    from qdrant_client import QdrantClient
    from config import QDRANT_URL, QDRANT_API_KEY
    cl = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    chunks, off = [], None
    while True:
        pts, off = cl.scroll("mutawwif_v4", limit=500, offset=off, with_payload=True, with_vectors=False)
        for p in pts:
            chunks.append((str(p.id), p.payload.get("metadata", {}).get("source", ""), p.payload.get("page_content", "")))
        if off is None:
            break
    print(f"[*] {len(chunks)} chunk dari mutawwif_v4")
    nchunks = [(i, src, norm(t)) for i, src, t in chunks]

    from data_loader import load_documents
    docs = load_documents(str(ROOT / "Dataset"))
    perfile = defaultdict(str)
    for d in docs:
        perfile[Path(d.metadata.get("source", "")).name] += "\n" + d.page_content
    nfiles = {k: norm(v) for k, v in perfile.items()}

    out = []
    for soal, ents in ENTITAS.items():
        for label, variants in ents.items():
            needles = sorted({norm_needle(v) for v in variants})
            hits = [(i, src) for i, src, t in nchunks if any(n in t for n in needles)]
            occ = sum(sum(t.count(n) for n in needles) for _, _, t in nchunks)
            fhits = {f: sum(t.count(n) for n in needles) for f, t in nfiles.items()}
            fhits = {f: c for f, c in fhits.items() if c}
            out.append(dict(soal=soal, entitas=label, varian=variants, ada_di_qdrant=bool(hits),
                            n_chunk=len(hits), n_kemunculan_chunk=occ,
                            chunk_ids=[h[0] for h in hits[:15]],
                            sumber_chunk=sorted({Path(h[1]).name for h in hits}),
                            ada_di_dataset=bool(fhits), kemunculan_per_file=fhits))
            print(f"soal {soal:>8} | {label:<22} | qdrant: {len(hits):>3} chunk | dataset: {fhits}")
    dst = ROOT / "revisi_r1/results/E7_entities"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "entity_search.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
