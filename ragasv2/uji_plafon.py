"""
Uji plafon: mengukur dampak query expansion & hybrid search TANPA judge LLM
============================================================================
Menjalankan RAGAS penuh butuh ~2,8 jam dan biaya API. Skrip ini memperkirakan
dampak perubahan retrieval dalam hitungan menit, dengan mengukur seberapa
banyak isi ground truth yang benar-benar tercakup di konteks yang terambil.

Ukurannya adalah proksi untuk context_recall: proporsi kata-isi (content word)
pada ground truth yang muncul di gabungan konteks top-k. Nilainya tidak sama
dengan skor RAGAS, tapi arah dan besaran perubahannya terbukti sejalan —
uji serupa sebelum reindex memprediksi kenaikan recall dengan tepat.

Empat konfigurasi dibandingkan:
    A. dense  + query asli      <- baseline saat ini (skor RAGAS 0,393)
    B. dense  + query diperluas <- dampak langkah 1 saja
    C. hybrid + query asli      <- dampak langkah 2 saja
    D. hybrid + query diperluas <- gabungan keduanya

Pakai:
    cd ragasv2
    ..\\.venv\\Scripts\\python.exe -u uji_plafon.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

from config import QDRANT_URL, QDRANT_API_KEY, RETRIEVE_K, RERANK_K
from vector_manager import get_embeddings
from chat_engine_v2 import get_reranker
from query_expansion import perluas
from reindex_hybrid import SUMBER as KOLEKSI_DENSE, TUJUAN as KOLEKSI_HYBRID, NAMA_DENSE, NAMA_SPARSE

DATASET = "pertanyaan_dataset.json"

# Kata fungsi dibuang agar yang diukur adalah istilah bermakna, bukan "yang/dan/di".
STOPWORD = set("""yang dan di ke dari itu ini pada untuk dengan tidak adalah akan atau
juga dalam oleh sebagai bahwa karena jika maka telah sudah bisa dapat harus ada
tersebut para saat setelah sebelum antara tentang seperti agar supaya namun tetapi
kepada dia ia mereka kami kita saya anda orang hal lain masih hanya lebih sangat
semua setiap wajib boleh tanpa bagi maupun serta yaitu adapun demikian apa bagaimana
kalau jadi kalo nya pun begitu sini situ""".split())


def kata_isi(teks: str) -> set:
    return {w for w in re.findall(r"[a-z']{4,}", teks.lower()) if w not in STOPWORD}


def cakupan(reference: str, konteks: list) -> float:
    ref = kata_isi(reference)
    if not ref:
        return 0.0
    ctx = kata_isi(" ".join(konteks))
    return len(ref & ctx) / len(ref)


def ambil(vs, reranker, query_cari: str, query_rerank: str) -> list:
    """Retrieve RETRIEVE_K lalu rerank ke RERANK_K, meniru pipeline app.py.

    query_cari  dipakai untuk retrieval (boleh sudah diperluas).
    query_rerank dipakai untuk reranking — selalu query ASLI, karena cross-encoder
    menilai kecocokan terhadap pertanyaan sebenarnya, bukan terhadap daftar istilah.
    """
    docs = vs.similarity_search(query_cari, k=RETRIEVE_K)
    if not docs:
        return []
    skor = reranker.predict([(query_rerank, d.page_content) for d in docs])
    urut = sorted(zip(docs, skor), key=lambda x: x[1], reverse=True)
    return [d.page_content for d, _ in urut[:RERANK_K]]


def main():
    data = json.loads(Path(DATASET).read_text(encoding="utf-8"))
    pertanyaan = data["questions"]
    referensi = [g[0] if isinstance(g, list) else g for g in data["ground_truths"]]
    print(f"[*] Dataset: {len(pertanyaan)} pertanyaan")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    emb = get_embeddings()
    reranker = get_reranker()

    vs_dense = QdrantVectorStore(
        client=client, collection_name=KOLEKSI_DENSE, embedding=emb,
    )

    vs_hybrid = None
    if client.collection_exists(KOLEKSI_HYBRID):
        vs_hybrid = QdrantVectorStore(
            client=client, collection_name=KOLEKSI_HYBRID, embedding=emb,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=NAMA_DENSE, sparse_vector_name=NAMA_SPARSE,
        )
    else:
        print(f"[!] Koleksi '{KOLEKSI_HYBRID}' belum ada — konfigurasi C dan D dilewati.")
        print(f"    Jalankan reindex_hybrid.py lebih dulu.\n")

    konfig = [("A. dense  + asli", vs_dense, False), ("B. dense  + diperluas", vs_dense, True)]
    if vs_hybrid is not None:
        konfig += [("C. hybrid + asli", vs_hybrid, False), ("D. hybrid + diperluas", vs_hybrid, True)]

    hasil = {}
    for nama, vs, pakai_perluas in konfig:
        print(f"[*] {nama} ...")
        skor = []
        for q, ref in zip(pertanyaan, referensi):
            qc = perluas(q, mode="glosarium") if pakai_perluas else q
            skor.append(cakupan(ref, ambil(vs, reranker, qc, q)))
        hasil[nama] = skor
        print(f"    rata-rata cakupan = {sum(skor)/len(skor):.3f}")
    print()

    dasar = hasil[konfig[0][0]]
    print("=" * 74)
    print(f"{'Konfigurasi':<24}{'Cakupan':>10}{'vs A':>10}{'nol':>7}{'membaik':>10}")
    print("-" * 74)
    for nama, _, _ in konfig:
        s = hasil[nama]
        rata = sum(s) / len(s)
        naik = sum(1 for a, b in zip(s, dasar) if a > b + 0.01)
        nol = sum(1 for x in s if x == 0)
        beda = "—" if nama == konfig[0][0] else f"{rata - sum(dasar)/len(dasar):+.3f}"
        print(f"{nama:<24}{rata:>10.3f}{beda:>10}{nol:>7}{naik:>10}")
    print("=" * 74)
    print()

    # Pertanyaan ber-recall nol pada evaluasi RAGAS terakhir. Kalau konfigurasi baru
    # mengangkat justru kelompok ini, dampaknya pada skor akhir akan paling terasa.
    nol_ragas = [4, 5, 6, 8, 9, 13, 14, 17, 20, 24, 27, 31]
    print("Fokus pada 12 pertanyaan yang context_recall-nya 0,00 di RAGAS terakhir:")
    for nama, _, _ in konfig:
        s = [hasil[nama][i] for i in nol_ragas if i < len(hasil[nama])]
        print(f"  {nama:<24} cakupan = {sum(s)/len(s):.3f}")

    Path("uji_plafon_hasil.json").write_text(
        json.dumps({k: v for k, v in hasil.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n[✔] Rincian per pertanyaan disimpan ke uji_plafon_hasil.json")


if __name__ == "__main__":
    main()
