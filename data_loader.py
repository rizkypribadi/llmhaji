import os
import glob
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from chunking_strategies import split_documents, ChunkingStrategy  # noqa: F401 — re-export

def load_documents(docs_dir):
    """
    Memuat dokumen (PDF, MD, JSON) dari direktori yang ditentukan.
    """
    print(f"[*] Memuat dokumen dari direktori: {docs_dir}")
    documents = []

    # 1. Load PDF files
    for pdf_file in glob.glob(os.path.join(docs_dir, "*.pdf")):
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"|-- Berhasil memuat PDF: {os.path.basename(pdf_file)} ({len(docs)} halaman)")
        except Exception as e:
            print(f"|-- Gagal memuat PDF {os.path.basename(pdf_file)}: {e}")

    # 2. Load Markdown files
    for md_file in glob.glob(os.path.join(docs_dir, "*.md")):
        try:
            loader = TextLoader(md_file, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
            print(f"|-- Berhasil memuat Markdown: {os.path.basename(md_file)} ({len(docs)} dokumen)")
        except Exception as e:
            print(f"|-- Gagal memuat Markdown {os.path.basename(md_file)}: {e}")

    # 3. Load JSON files (Crawl results)
    for json_file in glob.glob(os.path.join(docs_dir, "*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                content = data.get("content", "")
                metadata = data.get("metadata", {})
                metadata['source'] = json_file
                if content:
                    documents.append(Document(page_content=content, metadata=metadata))
            print(f"|-- Berhasil memuat JSON: {os.path.basename(json_file)}")
        except Exception as e:
            print(f"|-- Gagal memuat JSON {os.path.basename(json_file)}: {e}")

    print(f"[*] Total halaman/dokumen yang berhasil dimuat: {len(documents)}")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200, strategy="recursive", **kwargs):
    """
    Wrapper kompatibel-mundur untuk chunking_strategies.split_documents.
    Secara default menggunakan strategi "recursive" agar tidak mempengaruhi kode yang sudah ada.

    Args:
        documents    : List[Document]
        chunk_size   : Ukuran chunk (digunakan oleh strategi recursive & document_aware)
        chunk_overlap: Overlap antar chunk
        strategy     : Nama strategi — "recursive" | "document_aware" | "page_level" |
                       "hierarchical" | "semantic" | "agentic"
        **kwargs     : Parameter tambahan yang diteruskan ke strategi yang dipilih
    """
    from chunking_strategies import split_documents as _split
    print("[*] Memulai proses chunking dokumen...")
    result = _split(
        documents,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    if isinstance(result, dict):
        # Hierarchical → kembalikan child chunks agar kode lama tetap berjalan
        chunks = result["child"]
    else:
        chunks = result
    print(f"[*] Dokumen berhasil dibagi menjadi {len(chunks)} chunks.")
    return chunks
