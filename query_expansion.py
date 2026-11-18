"""
Query Expansion: jembatan bahasa awam -> istilah fiqih
=======================================================
Pertanyaan jemaah ditulis dengan bahasa sehari-hari ("Kalo saya kentut saat
ihram..."), sedangkan korpus memakai bahasa fiqih klasik ("hadats kecil
membatalkan wudhu, syarat sah thawaf"). Embedding harus menjembatani jurang
kosakata itu sendirian, dan sering gagal — terlihat dari 12 dari 33 pertanyaan
yang context_recall-nya 0,00 pada evaluasi korpus mutawwif_v2.

Modul ini memperkaya query dengan istilah teknis SEBELUM retrieval, tanpa
mengubah indeks sama sekali.

Tiga mode:
    glosarium : pemetaan kata kunci deterministik. Cepat, gratis, reprodusibel.
    llm       : model bahasa menuliskan ulang query ke istilah fiqih. Lebih
                luwes untuk kasus yang tak terduga, tapi lambat dan berbiaya.
    gabung    : glosarium + llm (default untuk hasil terbaik).

Hasil mode llm di-cache ke berkas JSON supaya evaluasi bisa diulang tanpa
memanggil API lagi — penting agar angka penelitian dapat direproduksi.
"""

import json
import re
from pathlib import Path

CACHE_FILE = Path(__file__).resolve().parent / "query_expansion_cache.json"

# Pemetaan: pola pada pertanyaan awam -> istilah fiqih yang dipakai di korpus.
# Kunci dicocokkan sebagai substring huruf kecil. Daftar ini disusun dari
# analisis pertanyaan ber-recall nol; tambahkan entri baru bila menemukan
# kegagalan retrieval lain.
GLOSARIUM = {
    "kentut": "hadats kecil membatalkan wudhu bersuci thaharah",
    "buang angin": "hadats kecil membatalkan wudhu bersuci thaharah",
    "netes air kencing": "beser salas al-baul inkontinensia uzur hadats terus-menerus",
    "beser": "salas al-baul hadats terus-menerus uzur",
    "pampers": "pakaian berjahit mukhith larangan ihram uzur",
    "popok": "pakaian berjahit mukhith larangan ihram",
    "celana dalam": "pakaian berjahit mukhith larangan ihram fidyah",
    "bersentuhan": "menyentuh perempuan ajnabi bukan mahram membatalkan wudhu talfiq madzhab",
    "lawan jenis": "ajnabi bukan mahram menyentuh kulit membatalkan wudhu",
    "haid": "haidh nifas thawaf ifadhah wanita bersuci",
    "menghajikan": "badal haji niyabah haji atas nama orang lain",
    "sudah meninggal": "badal haji niyabah orang yang telah wafat",
    "diwakilkan": "istinabah niyabah wakil uzur syar'i",
    "gundul": "halq mencukur rambut taqshir memendekkan tahallul",
    "cukur": "halq taqshir tahallul mencukur rambut",
    "botak": "halq taqshir tahallul isyarat pisau cukur",
    "lupa": "sahw lupa tidak tahu tidak sengaja fidyah dam gugur kewajiban",
    "tidak sengaja": "sahw jahil khatha' gugur kewajiban dam fidyah",
    "niat": "niyyah ihram talbiyah rukun haji",
    "bacaan wajib": "talbiyah niyyah rukun wajib haji lafazh",
    "kurang putaran": "thawaf tujuh putaran syawth batal mengulang",
    "batal": "batal fasid rusak mengulang qadha dam",
    "denda": "dam fidyah kafarat hadyu",
    "bayar kambing": "dam hadyu menyembelih kambing fidyah",
    "rebutan": "berdesakan mudarat menyakiti mafsadah sunnah istilam",
    "cium hajar aswad": "istilam hajar aswad sunnah thawaf",
    "murur": "murur Muzdalifah mabit lansia uzur hifzhun nafs",
    "tanazul": "tanazul mabit Mina uzur syar'i hifzhun nafs",
    "nafar": "nafar awal nafar tsani mabit Mina tasyriq",
    "wukuf": "wukuf Arafah rukun haji waktu wukuf",
    "melontar": "melontar jumrah ramyu jamrah aqabah tasyriq",
    "keluar dari mekkah": "miqat ihram masuk tanah haram umrah berulang",
    "tahiyyatul masjid": "tahiyyatul masjid thawaf qudum Masjidil Haram sunnah",
    "meninggal": "wafat jenazah ihram badal haji dikafani",
    "thawaf wada": "thawaf wada' meninggalkan Makkah wajib dam",
}

PROMPT_LLM = """Anda ahli fiqih haji dan umrah. Tugas Anda BUKAN menjawab pertanyaan.

Tulis ulang pertanyaan jemaah berikut menjadi satu baris kata kunci istilah fiqih
klasik yang kemungkinan besar muncul di kitab manasik (Al-Majmu', Al-Mughni,
Fiqih Nawawi, Tuntunan Manasik Kemenag).

Aturan:
- Keluarkan HANYA deretan istilah, dipisahkan spasi. Tanpa kalimat, tanpa penjelasan.
- Gunakan istilah Arab yang lazim ditulis di kitab terjemahan (mis. hadats, thawaf,
  ihram, dam, fidyah, ajnabi, badal, mabit, talfiq, uzur syar'i).
- Maksimal 25 kata.

Pertanyaan jemaah: {q}

Istilah fiqih:"""


def _muat_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _simpan_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def perluas_glosarium(query: str) -> str:
    """Istilah tambahan dari pemetaan deterministik. Kembalikan string kosong bila tak ada yang cocok."""
    q = query.lower()
    tambahan = []
    for pola, istilah in GLOSARIUM.items():
        if pola in q:
            tambahan.extend(t for t in istilah.split() if t.lower() not in q)
    # Buang duplikat, pertahankan urutan agar hasilnya deterministik.
    unik = list(dict.fromkeys(tambahan))
    return " ".join(unik)


def perluas_llm(query: str, client, model: str, cache: dict) -> str:
    """Perluasan via LLM, dengan cache agar pemanggilan berulang tidak berbiaya."""
    kunci = f"{model}::{query}"
    if kunci in cache:
        return cache[kunci]

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT_LLM.format(q=query)}],
        # Anggaran besar: model reasoning memotong reasoning_tokens dari max_tokens,
        # sehingga nilai kecil membuat content kembali kosong.
        max_tokens=4096,
        temperature=0.0,
    )
    teks = (resp.choices[0].message.content or "").strip()
    # Ambil baris berisi pertama; sebagian model menambahi baris pengantar.
    baris = [b.strip() for b in teks.splitlines() if b.strip()]
    hasil = baris[-1] if baris else ""
    hasil = re.sub(r"^[-*\d.\s]+", "", hasil)[:400]

    cache[kunci] = hasil
    _simpan_cache(cache)
    return hasil


def perluas(query: str, mode: str = "gabung", client=None, model: str = None, cache: dict = None) -> str:
    """
    Kembalikan query yang sudah diperkaya. Query asli SELALU dipertahankan di depan
    supaya sinyal aslinya tidak hilang — istilah tambahan hanya melengkapi.
    """
    bagian = [query]

    if mode in ("glosarium", "gabung"):
        g = perluas_glosarium(query)
        if g:
            bagian.append(g)

    if mode in ("llm", "gabung"):
        if client is None or model is None:
            raise ValueError("mode 'llm'/'gabung' membutuhkan client dan model")
        if cache is None:
            cache = _muat_cache()
        l = perluas_llm(query, client, model, cache)
        if l:
            bagian.append(l)

    return " ".join(bagian)


if __name__ == "__main__":
    contoh = [
        "Kalo saya kentut saat saya dalam keadaan ihram bagaimana hukumnya?",
        "Saat thawaf saya bersentuhan dengan lawan jenis, apakah thawaf saya batal?",
        "Saya ingin menghajikan orang tua saya yang sudah meninggal, apakah bisa?",
        "Kondisi saya ini sering netes air kencing, apakah boleh saya menggunakan pampers saat ihram?",
    ]
    print("Mode glosarium (tanpa API):\n")
    for q in contoh:
        print(f"  ASLI     : {q}")
        print(f"  DIPERLUAS: {perluas(q, mode='glosarium')}\n")
