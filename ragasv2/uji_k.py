"""
Menyapu nilai RERANK_K untuk mencari titik terbaik
===================================================
Profil metrik menunjukkan precision (0,795) jauh di atas recall (0,393): yang
terambil memang relevan, tapi jumlahnya kurang. Artinya ada ruang menukar
sedikit precision demi recall dengan memperbesar RERANK_K.

Berapa besar? Itu harus diukur, bukan ditebak. Skrip ini mengukur cakupan
ground truth pada beberapa nilai k memakai konfigurasi retrieval terbaik,
tanpa memanggil judge LLM.

Perlu diingat saat membaca hasilnya: cakupan HAMPIR SELALU naik seiring k,
karena konteks makin panjang. Yang dicari bukan k dengan cakupan tertinggi,
melainkan k tempat kenaikannya mulai mendatar — setelah itu tambahan dokumen
hanya mengencerkan konteks dan menekan precision.

Pakai:
    cd ragasv2
    ..\\.venv\\Scripts\\python.exe -u uji_k.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

from config import QDRANT_URL, QDRANT_API_KEY, RETRIEVE_K
from vector_manager import get_embeddings
from chat_engine_v2 import get_reranker
from query_expansion import perluas
from reindex_hybrid import TUJUAN as KOLEKSI_HYBRID, SUMBER as KOLEKSI_DENSE, NAMA_DENSE, NAMA_SPARSE
from uji_plafon import cakupan, DATASET

NILAI_K = [3, 5, 8, 10, 15]


def main():
    data = json.loads(Path(DATASET).read_text(encoding="utf-8"))
    pertanyaan = data["questions"]
    referensi = [g[0] if isinstance(g, list) else g for g in data["ground_truths"]]

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    emb = get_embeddings()
    reranker = get_reranker()

    if client.collection_exists(KOLEKSI_HYBRID):
        vs = QdrantVectorStore(
            client=client, collection_name=KOLEKSI_HYBRID, embedding=emb,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=NAMA_DENSE, sparse_vector_name=NAMA_SPARSE,
        )
        label = "hybrid + query diperluas"
    else:
        vs = QdrantVectorStore(client=client, collection_name=KOLEKSI_DENSE, embedding=emb)
        label = "dense + query diperluas"

    print(f"[*] Konfigurasi: {label}")
    print(f"[*] Retrieve {RETRIEVE_K} kandidat, disaring ke k = {NILAI_K}\n")

    # Retrieval dan reranking dihitung SEKALI per pertanyaan, lalu dipotong pada
    # tiap nilai k. Menjalankan ulang reranker untuk setiap k hanya membuang waktu
    # karena urutannya identik.
    urutan = []
    for i, q in enumerate(pertanyaan, 1):
        docs = vs.similarity_search(perluas(q, mode="glosarium"), k=RETRIEVE_K)
        skor = reranker.predict([(q, d.page_content) for d in docs])
        rank = [d.page_content for d, _ in sorted(zip(docs, skor), key=lambda x: x[1], reverse=True)]
        urutan.append(rank)
        print(f"|-- {i}/{len(pertanyaan)}")
    print()

    nol_ragas = [4, 5, 6, 8, 9, 13, 14, 17, 20, 24, 27, 31]

    print("=" * 62)
    print(f"{'k':>4}{'cakupan':>11}{'selisih':>11}{'12 soal nol':>14}{'panjang ctx':>14}")
    print("-" * 62)
    sebelum = None
    for k in NILAI_K:
        skor = [cakupan(ref, rank[:k]) for ref, rank in zip(referensi, urutan)]
        rata = sum(skor) / len(skor)
        fokus = sum(skor[i] for i in nol_ragas if i < len(skor)) / len(nol_ragas)
        panjang = sum(sum(len(c) for c in rank[:k]) for rank in urutan) // len(urutan)
        delta = "—" if sebelum is None else f"{rata - sebelum:+.3f}"
        print(f"{k:>4}{rata:>11.3f}{delta:>11}{fokus:>14.3f}{panjang:>14}")
        sebelum = rata
    print("=" * 62)
    print()
    print("Pilih k terkecil yang kenaikannya masih berarti. Setelah kenaikan")
    print("melandai (< 0,01), menambah k hanya menekan context_precision.")


if __name__ == "__main__":
    main()
