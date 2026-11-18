"""
Membangun koleksi hybrid (dense + sparse BM25) dari koleksi dense yang sudah ada
================================================================================
Metrik context_entity_recall adalah yang paling lemah (0,159). Entitas — nama
ulama, istilah Arab, angka, nama tempat — justru paling mudah ditangkap
pencarian leksikal dan paling mudah dikaburkan embedding padat. Hybrid search
menggabungkan keduanya: BM25 menangkap kecocokan kata persis, bge-m3 menangkap
kedekatan makna.

Vektor dense TIDAK dihitung ulang. Skrip ini menyalinnya dari mutawwif_v2 dan
hanya menambahkan vektor sparse, yang perhitungannya murni CPU dan cepat.
Menghitung ulang embedding dense akan memakan ~2,4 jam tanpa manfaat apa pun,
karena model embedding-nya sama persis.

Pakai:
    ..\\.venv\\Scripts\\python.exe -u reindex_hybrid.py --uji     # validasi 50 titik dulu
    ..\\.venv\\Scripts\\python.exe -u reindex_hybrid.py           # jalan penuh
"""

import argparse
import sys

from qdrant_client import QdrantClient, models
from config import QDRANT_URL, QDRANT_API_KEY

SUMBER = "mutawwif_v2"
TUJUAN = "mutawwif_v3"
TUJUAN_UJI = "mutawwif_v3_uji"

# Harus sama dengan default langchain_qdrant.QdrantVectorStore:
#   vector_name=''  (dense tanpa nama, seperti mutawwif_v2)
#   sparse_vector_name='langchain-sparse'
# Kalau tidak cocok, QdrantVectorStore menolak koleksi saat validasi.
NAMA_DENSE = ""
NAMA_SPARSE = "langchain-sparse"

BATCH = 128


def siapkan_koleksi(client: QdrantClient, nama: str, dim: int) -> None:
    if client.collection_exists(nama):
        print(f"[*] Menghapus koleksi lama '{nama}'...")
        client.delete_collection(nama)

    print(f"[*] Membuat koleksi '{nama}' (dense {dim}d + sparse BM25)...")
    client.create_collection(
        collection_name=nama,
        vectors_config={
            NAMA_DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)
        },
        # modifier=IDF wajib untuk model Qdrant/bm25: bobot IDF dihitung di sisi
        # server berdasarkan statistik koleksi, bukan dibawa oleh vektornya.
        sparse_vectors_config={
            NAMA_SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uji", action="store_true",
                    help="hanya proses 50 titik ke koleksi uji, lalu verifikasi pencarian hybrid")
    args = ap.parse_args()

    try:
        from langchain_qdrant import FastEmbedSparse
    except ImportError:
        print("[!] Paket 'fastembed' belum terpasang. Jalankan:")
        print("    .\\.venv\\Scripts\\python.exe -m pip install fastembed")
        sys.exit(1)

    tujuan = TUJUAN_UJI if args.uji else TUJUAN
    batas = 50 if args.uji else None

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    info = client.get_collection(SUMBER)
    vcfg = info.config.params.vectors
    dim = next(iter(vcfg.values())).size if isinstance(vcfg, dict) else vcfg.size
    total = info.points_count
    print(f"[*] Sumber : {SUMBER} ({total} titik, dense {dim}d)")
    print(f"[*] Tujuan : {tujuan}")
    print()

    print("[*] Memuat model sparse BM25 (Qdrant/bm25)...")
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")

    siapkan_koleksi(client, tujuan, dim)

    diproses = 0
    offset = None
    while True:
        titik, offset = client.scroll(
            SUMBER, limit=BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not titik:
            break

        teks = [(p.payload or {}).get("page_content", "") for p in titik]
        vek_sparse = sparse.embed_documents(teks)

        upsert = []
        for p, sv in zip(titik, vek_sparse):
            dense_vec = p.vector
            # Koleksi sumber memakai vektor tanpa nama; qdrant-client
            # mengembalikannya sebagai list, bukan dict.
            if isinstance(dense_vec, dict):
                dense_vec = dense_vec.get(NAMA_DENSE) or next(iter(dense_vec.values()))
            upsert.append(models.PointStruct(
                id=p.id,
                vector={
                    NAMA_DENSE: dense_vec,
                    NAMA_SPARSE: models.SparseVector(indices=sv.indices, values=sv.values),
                },
                payload=p.payload,
            ))

        client.upsert(collection_name=tujuan, points=upsert, wait=True)
        diproses += len(upsert)
        print(f"|-- {diproses}/{batas or total} titik")

        if batas and diproses >= batas:
            break
        if offset is None:
            break

    print()
    hasil = client.get_collection(tujuan)
    print(f"[✔] Selesai. '{tujuan}' berisi {hasil.points_count} titik.")

    if args.uji:
        print()
        print("[*] Verifikasi pencarian hybrid pada koleksi uji...")
        from langchain_qdrant import QdrantVectorStore, RetrievalMode
        from vector_manager import get_embeddings

        vs = QdrantVectorStore(
            client=client,
            collection_name=tujuan,
            embedding=get_embeddings(),
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=NAMA_DENSE,
            sparse_vector_name=NAMA_SPARSE,
        )
        docs = vs.similarity_search("dam fidyah kafarat", k=3)
        for i, d in enumerate(docs, 1):
            print(f"  {i}. {' '.join(d.page_content.split())[:90]}")
        print()
        print("[✔] Pencarian hybrid berfungsi. Jalankan tanpa --uji untuk koleksi penuh.")
        print(f"    Koleksi uji '{tujuan}' boleh dihapus setelah ini.")


if __name__ == "__main__":
    main()
