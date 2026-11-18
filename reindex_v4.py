"""
Script Re-indexing: Dataset/ -> mutawwif_v4 (Qdrant lokal)
============================================================
Koleksi baru, terpisah dari mutawwif_v2/v3. Dibuat dari isi Dataset/
saat ini, yang sudah termasuk wawancara pakar.md (dibersihkan) dan
OCR "02 Panduan Lengkap Haji & Umroh Syaikh Imam Nawawi.md".
"""

import sys
from config import EMBEDDING_MODEL, DATASET_DIR
from data_loader import load_documents, split_documents
from vector_manager import get_another_vector_store, add_documents_in_batches

TARGET_COLLECTION = "mutawwif_v4"


def main():
    print("=" * 70)
    print("  RE-INDEXING DATASET -> QDRANT LOKAL (mutawwif_v4)")
    print("=" * 70)
    print()
    print(f"Embedding Model : {EMBEDDING_MODEL}")
    print(f"Koleksi tujuan  : {TARGET_COLLECTION} (baru)")
    print(f"Dokumen         : {DATASET_DIR}/")
    print()

    try:
        print("[1/3] Menyiapkan koleksi tujuan...")
        # force_reindex=False: koleksi sudah dibuat kosong di percobaan sebelumnya
        # (delete-lalu-buat-ulang berisiko kena I/O timeout di filesystem WSL2 ini).
        vector_store = get_another_vector_store(
            force_reindex=False, collection_name=TARGET_COLLECTION
        )

        print()
        print("[2/3] Memuat dokumen dari direktori...")
        documents = load_documents(DATASET_DIR)

        if not documents:
            print(f"[!] Tidak ada dokumen ditemukan di folder {DATASET_DIR}/")
            sys.exit(1)

        print()
        print("[3/3] Memecah dokumen menjadi chunks (strategi: recursive)...")
        chunks = split_documents(documents, strategy="recursive")

        print()
        print("[*] Mengindex chunks ke Qdrant Vector Database...")
        add_documents_in_batches(vector_store, chunks, batch_size=10)

        print()
        print("=" * 70)
        print("RE-INDEXING SELESAI!")
        print("=" * 70)
        print(f"Total dokumen/halaman: {len(documents)}")
        print(f"Total chunks         : {len(chunks)}")
        print(f"Koleksi              : {TARGET_COLLECTION}")

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR SAAT RE-INDEXING")
        print("=" * 70)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
