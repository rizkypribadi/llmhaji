import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from config import QDRANT_URL, QDRANT_API_KEY, CLUSTER_NAME, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

# Mapping dimensi vektor untuk setiap model
EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-minilm-l6-v2": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-m3": 1024,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}

# Model dengan nama-nama berikut dianggap OpenAI-compatible (diakses via sumopod)
OPENAI_EMBEDDING_PREFIXES = ("text-embedding-",)


def is_openai_embedding_model(model_name: str) -> bool:
    return model_name.startswith(OPENAI_EMBEDDING_PREFIXES)


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """
    Inisialisasi embedding model, otomatis pilih backend:
    - OpenAI-compatible (text-embedding-*) → via sumopod (OPENAI_BASE_URL)
    - Selain itu → HuggingFace (lokal, device CPU)
    """
    if is_openai_embedding_model(model_name):
        return OpenAIEmbeddings(
            model=model_name,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True},
    )

def get_embedding_dimension(model_name: str) -> int:
    """
    Dapatkan dimensi vektor untuk model embedding yang digunakan.
    """
    if model_name in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[model_name]
    else:
        # Fallback: coba load model dan deteksi dimensi
        print(f"[!] Dimensi untuk {model_name} tidak diketahui, detecting...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            dim = model.get_sentence_embedding_dimension()
            print(f"|-- Detected dimension: {dim}")
            return dim
        except Exception as e:
            print(f"[!] Error detecting dimension: {e}")
            return 768  # Default fallback

def get_vector_store(force_reindex=False):
    """
    Inisialisasi Vector Store (Qdrant). 
    
    Args:
        force_reindex: Jika True, hapus koleksi existing dan buat ulang.
                       WAJIB True jika ganti embedding model!
    """
    print("[*] Inisialisasi Vector Store (Qdrant)...")
    print(f"|-- Embedding Model: {EMBEDDING_MODEL}")
    
    # Inisialisasi model embedding
    embeddings = get_embeddings(EMBEDDING_MODEL)
    
    # Dapatkan dimensi vektor yang sesuai
    vector_dim = get_embedding_dimension(EMBEDDING_MODEL)
    print(f"|-- Vector Dimension: {vector_dim}")
    
    # Inisialisasi Qdrant Client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # client.delete_collection(CLUSTER_NAME)
    
    # Cek koleksi
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if force_reindex and CLUSTER_NAME in collection_names:
        print(f"[*] Menghapus koleksi '{CLUSTER_NAME}' untuk re-indexing ulang...")
        client.delete_collection(CLUSTER_NAME)
        collection_names.remove(CLUSTER_NAME)

    if CLUSTER_NAME not in collection_names:
        print(f"[*] Koleksi '{CLUSTER_NAME}' belum ada. Membuat koleksi baru...")
        client.create_collection(
            collection_name=CLUSTER_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"[✔] Koleksi dibuat dengan dimensi {vector_dim}")
        # Vector store akan diisi oleh main.py jika diperlukan
    else:
        # Verifikasi dimensi koleksi existing
        collection_info = client.get_collection(CLUSTER_NAME)
        vectors_config = collection_info.config.params.vectors
        # qdrant-client baru: vectors bisa berupa dict (named vectors) atau VectorParams langsung
        if isinstance(vectors_config, dict):
            existing_dim = next(iter(vectors_config.values())).size
        else:
            existing_dim = vectors_config.size
        
        if existing_dim != vector_dim:
            print(f"[!] PERINGATAN: Dimensi koleksi ({existing_dim}) tidak sesuai dengan model ({vector_dim})")
            print(f"[!] Anda perlu menjalankan dengan force_reindex=True untuk membuat ulang koleksi!")
            raise ValueError(
                f"Dimensi tidak cocok: koleksi={existing_dim}, model={vector_dim}. "
                f"Jalankan dengan force_reindex=True atau ganti EMBEDDING_MODEL di config.py"
            )
        
        print(f"[*] Koleksi '{CLUSTER_NAME}' sudah tersedia (dimensi: {existing_dim}).")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=CLUSTER_NAME,
        embedding=embeddings,
    )
    
    return vector_store

def get_another_vector_store(force_reindex=False, collection_name=None):
    """
    Inisialisasi Vector Store (Qdrant). 
    
    Args:
        force_reindex: Jika True, hapus koleksi existing dan buat ulang.
                       WAJIB True jika ganti embedding model!
    """
    print("[*] Inisialisasi Vector Store (Qdrant)...")
    print(f"|-- Embedding Model: {EMBEDDING_MODEL}")
    
    # Inisialisasi model embedding
    embeddings = get_embeddings(EMBEDDING_MODEL)
    
    # Dapatkan dimensi vektor yang sesuai
    vector_dim = get_embedding_dimension(EMBEDDING_MODEL)
    print(f"|-- Vector Dimension: {vector_dim}")
    
    # Inisialisasi Qdrant Client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # client.delete_collection(CLUSTER_NAME)
    
    # Cek koleksi
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if force_reindex and collection_name in collection_names:
        print(f"[*] Menghapus koleksi '{collection_name}' untuk re-indexing ulang...")
        client.delete_collection(collection_name)
        collection_names.remove(collection_name)

    if collection_name not in collection_names:
        print(f"[*] Koleksi '{collection_name}' belum ada. Membuat koleksi baru...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )
        print(f"[✔] Koleksi dibuat dengan dimensi {vector_dim}")
        # Vector store akan diisi oleh main.py jika diperlukan
    else:
        # Verifikasi dimensi koleksi existing
        collection_info = client.get_collection(collection_name)
        vectors_config = collection_info.config.params.vectors
        # qdrant-client baru: vectors bisa berupa dict (named vectors) atau VectorParams langsung
        if isinstance(vectors_config, dict):
            existing_dim = next(iter(vectors_config.values())).size
        else:
            existing_dim = vectors_config.size
        
        if existing_dim != vector_dim:
            print(f"[!] PERINGATAN: Dimensi koleksi ({existing_dim}) tidak sesuai dengan model ({vector_dim})")
            print(f"[!] Anda perlu menjalankan dengan force_reindex=True untuk membuat ulang koleksi!")
            raise ValueError(
                f"Dimensi tidak cocok: koleksi={existing_dim}, model={vector_dim}. "
                f"Jalankan dengan force_reindex=True atau ganti EMBEDDING_MODEL di config.py"
            )
        
        print(f"[*] Koleksi '{collection_name}' sudah tersedia (dimensi: {existing_dim}).")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    return vector_store    

def add_documents_in_batches(vector_store, splits, batch_size=10):
    """
    Menambahkan chunks dokumen ke Qdrant dalam batch kecil untuk kestabilan.
    """
    print("[*] Menyimpan chunks ke dalam Qdrant Vector DB...")
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                print(f"|-- Memasukkan batch {i} - {i + len(batch)}")
                vector_store.add_documents(batch)
                success = True
                time.sleep(0.5) 
            except Exception as e:
                print(f"|-- GAGAL batch {i}, mencoba ulang... ({retries} sisa). Error: {e}")
                retries -= 1
                time.sleep(2)
        if not success:
            print(f"|-- [ERROR] GAGAL memasukkan batch {i} setelah 3 kali percobaan.")
    
    print("[*] Proses indexing ke Qdrant selesai!")
