"""
chunking_strategies.py
======================
Modul chunking yang menyediakan berbagai strategi pemecahan dokumen untuk pipeline RAG.

Strategi yang tersedia:
  - recursive       : RecursiveCharacterTextSplitter (default, general-purpose)
  - document_aware  : Parsing berbasis struktur dokumen (Markdown header / HTML header)
  - page_level      : Satu chunk per halaman (ideal untuk PDF yang sudah dipisah per halaman)
  - hierarchical    : Dua lapis chunk — parent (besar) + child (kecil) dengan metadata linking
  - semantic        : Pemecahan berbasis kesamaan semantik antar kalimat (butuh embedding model)
  - agentic         : Pemecahan berbasis LLM — model memutuskan batas chunk secara kontekstual
"""

from __future__ import annotations

import re
import json
from enum import Enum
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


# ---------------------------------------------------------------------------
# Enum strategi
# ---------------------------------------------------------------------------

class ChunkingStrategy(str, Enum):
    RECURSIVE = "recursive"
    DOCUMENT_AWARE = "document_aware"
    PAGE_LEVEL = "page_level"
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    AGENTIC = "agentic"


# ---------------------------------------------------------------------------
# 1. Recursive Character Text Splitting
# ---------------------------------------------------------------------------

def chunk_recursive(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    Memecah dokumen secara rekursif dengan daftar separator bertingkat.
    Separator default: ["\n\n", "\n", " ", ""].
    Cocok untuk teks umum tanpa struktur khusus.
    """
    print(f"[*] Chunking: RECURSIVE (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function = len,
        separators=separators or ["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[✔] RECURSIVE → {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# 2. Document-Aware Parsing (struktur dokumen)
# ---------------------------------------------------------------------------

def chunk_document_aware(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Memecah dokumen berdasarkan struktur logisnya:
      - Markdown (.md) → dipecah mengikuti hierarki heading (H1–H3)
      - HTML           → dipecah mengikuti tag heading HTML
      - Lainnya        → fallback ke Recursive

    Metadata heading ditambahkan ke setiap chunk.
    """
    print("[*] Chunking: DOCUMENT-AWARE")

    md_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    result: List[Document] = []
    for doc in documents:
        source = doc.metadata.get("source", "")

        if source.endswith(".md"):
            # Markdown: split by heading, lalu sub-split jika masih terlalu besar
            header_chunks = md_header_splitter.split_text(doc.page_content)
            for hc in header_chunks:
                hc.metadata.update(doc.metadata)
                if len(hc.page_content) > chunk_size:
                    sub = fallback_splitter.split_documents([hc])
                    result.extend(sub)
                else:
                    result.append(hc)

        elif source.endswith(".html") or source.endswith(".htm"):
            try:
                from langchain_text_splitters import HTMLHeaderTextSplitter
                html_splitter = HTMLHeaderTextSplitter(
                    headers_to_split_on=[
                        ("h1", "h1"),
                        ("h2", "h2"),
                        ("h3", "h3"),
                    ]
                )
                html_chunks = html_splitter.split_text(doc.page_content)
                for hc in html_chunks:
                    hc.metadata.update(doc.metadata)
                    if len(hc.page_content) > chunk_size:
                        sub = fallback_splitter.split_documents([hc])
                        result.extend(sub)
                    else:
                        result.append(hc)
            except ImportError:
                print("[!] HTMLHeaderTextSplitter tidak tersedia, fallback ke Recursive.")
                result.extend(fallback_splitter.split_documents([doc]))

        else:
            # Fallback untuk PDF, JSON, dll.
            result.extend(fallback_splitter.split_documents([doc]))

    print(f"[✔] DOCUMENT-AWARE → {len(result)} chunks")
    return result


# ---------------------------------------------------------------------------
# 3. Page-Level Chunking
# ---------------------------------------------------------------------------

def chunk_page_level(
    documents: List[Document],
    min_page_length: int = 50,
    **kwargs,  # Menyerap parameter tidak relevan seperti chunk_size/chunk_overlap
) -> List[Document]:
    """
    Memperlakukan setiap halaman sebagai satu chunk tersendiri.
    Ideal untuk PDF yang sudah dipisah per halaman oleh PyPDFLoader.
    Halaman dengan teks sangat pendek (< min_page_length karakter) dilewati.
    """
    print(f"[*] Chunking: PAGE-LEVEL (min_length={min_page_length})")
    result: List[Document] = []
    for doc in documents:
        text = doc.page_content.strip()
        if len(text) >= min_page_length:
            result.append(Document(page_content=text, metadata=doc.metadata.copy()))
        else:
            print(f"|-- Skip halaman pendek ({len(text)} karakter): {doc.metadata.get('source', '')}")
    print(f"[✔] PAGE-LEVEL → {len(result)} chunks")
    return result


# ---------------------------------------------------------------------------
# 4. Hierarchical / Parent-Child Chunking
# ---------------------------------------------------------------------------

def chunk_hierarchical(
    documents: List[Document],
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 50,
) -> dict[str, List[Document]]:
    """
    Membuat dua lapis chunk:
      - parent_chunks : chunk besar yang menyimpan konteks luas
      - child_chunks  : chunk kecil untuk retrieval presisi tinggi,
                        setiap child menyimpan `parent_id` di metadata-nya

    Return:
        {
            "parent": List[Document],   # simpan di docstore / full-text store
            "child":  List[Document],   # upsert ke vector store
        }

    Cara pakai:
        result = chunk_hierarchical(docs)
        add_to_vectorstore(result["child"])
        # Saat retrieval, gunakan parent_id di metadata child untuk fetch parent
    """
    print(
        f"[*] Chunking: HIERARCHICAL "
        f"(parent={parent_chunk_size}, child={child_chunk_size})"
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    parent_chunks: List[Document] = []
    child_chunks: List[Document] = []

    parent_splits = parent_splitter.split_documents(documents)
    for p_idx, parent in enumerate(parent_splits):
        parent_id = f"parent_{p_idx}"
        parent.metadata["chunk_id"] = parent_id
        parent.metadata["chunk_type"] = "parent"
        parent_chunks.append(parent)

        children = child_splitter.split_documents([parent])
        for c_idx, child in enumerate(children):
            child.metadata["parent_id"] = parent_id
            child.metadata["chunk_id"] = f"{parent_id}_child_{c_idx}"
            child.metadata["chunk_type"] = "child"
            child_chunks.append(child)

    print(f"[✔] HIERARCHICAL → {len(parent_chunks)} parent | {len(child_chunks)} child chunks")
    return {"parent": parent_chunks, "child": child_chunks}


# ---------------------------------------------------------------------------
# 5. Semantic Chunking
# ---------------------------------------------------------------------------

def chunk_semantic(
    documents: List[Document],
    embedding_model_name: str = "BAAI/bge-m3",
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
    **kwargs,  # Menyerap parameter tidak relevan seperti chunk_size/chunk_overlap
) -> List[Document]:
    """
    Memecah dokumen berdasarkan kesamaan semantik antar kalimat menggunakan embeddings.
    Batas chunk dibuat di titik di mana kesamaan semantik turun signifikan.

    Membutuhkan: langchain-experimental, sentence-transformers

    Args:
        embedding_model_name     : Nama model HuggingFace untuk embedding
        breakpoint_threshold_type: "percentile" | "standard_deviation" | "interquartile"
        breakpoint_threshold_amount: Threshold nilai (persentil jika tipe "percentile")
    """
    print(f"[*] Chunking: SEMANTIC (model={embedding_model_name}, threshold={breakpoint_threshold_type}@{breakpoint_threshold_amount})")
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        raise ImportError(
            "Semantic chunking membutuhkan 'langchain-experimental' dan 'langchain-huggingface'. "
            "Jalankan: pip install langchain-experimental langchain-huggingface"
        ) from e

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name, 
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    result: List[Document] = []
    for doc in documents:
        try:
            semantic_chunks = chunker.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata],
            )
            result.extend(semantic_chunks)
        except Exception as e:
            print(f"[!] Semantic chunking gagal untuk dokumen '{doc.metadata.get('source', '')}': {e}. Fallback ke recursive.")
            fallback = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200, 
                length_function=len,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            result.extend(fallback.split_documents([doc]))

    print(f"[✔] SEMANTIC → {len(result)} chunks")
    return result


# ---------------------------------------------------------------------------
# 6. Agentic Chunking
# ---------------------------------------------------------------------------

def chunk_agentic(
    documents: List[Document],
    client: Any = None,
    model_name: Optional[str] = None,
    max_propositions_per_doc: int = 50,
    fallback_chunk_size: int = 1000,
    fallback_chunk_overlap: int = 200,
    **kwargs,  # Menyerap parameter tidak relevan seperti chunk_size/chunk_overlap
) -> List[Document]:
    """
    Memecah dokumen menggunakan LLM yang memutuskan batas chunk secara kontekstual.
    Model mengekstrak 'proposisi' — pernyataan atomik mandiri yang merangkum fakta tunggal.

    Cocok untuk teks religius / fiqh yang membutuhkan pemahaman konteks mendalam.

    Args:
        client                  : Instance OpenAI (atau kompatibel). Jika None, dibuat otomatis.
        model_name              : Nama model LLM. Fallback ke OPENAI_MODEL di config.
        max_propositions_per_doc: Batas jumlah proposisi per dokumen untuk menghindari over-splitting.
        fallback_chunk_size     : Ukuran chunk jika LLM gagal.
        fallback_chunk_overlap  : Overlap chunk jika LLM gagal.

    Return:
        List[Document] — setiap proposisi sebagai chunk mandiri.
    """
    print("[*] Chunking: AGENTIC (LLM-driven proposition extraction)")

    # Siapkan LLM client
    if client is None:
        try:
            from openai import OpenAI
            from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            if model_name is None:
                model_name = OPENAI_MODEL
        except Exception as e:
            raise RuntimeError(
                "Agentic chunking membutuhkan OpenAI client yang valid. "
                "Pastikan OPENAI_API_KEY dan OPENAI_BASE_URL tersedia di config.py."
            ) from e

    system_prompt = (
        "Anda adalah asisten ekstraksi pengetahuan. "
        "Tugas Anda: baca teks berikut dan pecah menjadi proposisi-proposisi atomik. "
        "Setiap proposisi adalah satu pernyataan mandiri yang mengandung satu fakta utama. "
        "Proposisi harus bisa dipahami tanpa membaca teks asli. "
        "Kembalikan HANYA daftar JSON array of strings. Contoh: [\"Proposisi 1.\", \"Proposisi 2.\"]"
    )

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=fallback_chunk_size,
        chunk_overlap=fallback_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Pre-split menjadi potongan ukuran sedang agar tidak melebihi context window LLM
    pre_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    pre_chunks = pre_splitter.split_documents(documents)

    result: List[Document] = []
    for chunk_idx, pre_chunk in enumerate(pre_chunks):
        text = pre_chunk.page_content.strip()
        if not text:
            continue

        print(f"|-- Agentic chunking pre-chunk {chunk_idx + 1}/{len(pre_chunks)}...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Teks:\n{text}"},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

            # Ekstrak JSON array dari respons model
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not json_match:
                raise ValueError("Respons tidak mengandung JSON array yang valid.")

            propositions: List[str] = json.loads(json_match.group())

            # Batasi jumlah proposisi
            propositions = propositions[:max_propositions_per_doc]

            for prop in propositions:
                prop = prop.strip()
                if prop:
                    result.append(
                        Document(
                            page_content=prop,
                            metadata={
                                **pre_chunk.metadata,
                                "chunk_type": "proposition",
                                "pre_chunk_idx": chunk_idx,
                            },
                        )
                    )

        except Exception as e:
            print(f"[!] Agentic chunking gagal untuk chunk {chunk_idx + 1}: {e}. Fallback ke recursive.")
            result.extend(fallback_splitter.split_documents([pre_chunk]))

    print(f"[✔] AGENTIC → {len(result)} proposisi/chunks")
    return result


# ---------------------------------------------------------------------------
# Fungsi Utama: split_documents (unified interface)
# ---------------------------------------------------------------------------

def split_documents(
    documents: List[Document],
    strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
    **kwargs: Any,
) -> List[Document] | dict[str, List[Document]]:
    """
    Antarmuka tunggal untuk semua strategi chunking.

    Args:
        documents : List[Document] — dokumen hasil load_documents()
        strategy  : ChunkingStrategy enum atau string nama strategi
        **kwargs  : Parameter tambahan yang diteruskan ke fungsi strategi masing-masing

    Returns:
        - List[Document]  untuk semua strategi kecuali "hierarchical"
        - dict{"parent": [...], "child": [...]} untuk "hierarchical"

    Contoh pemakaian:
        from chunking_strategies import split_documents, ChunkingStrategy

        # Recursive (default)
        chunks = split_documents(docs)

        # Document-Aware
        chunks = split_documents(docs, strategy="document_aware")

        # Page-Level
        chunks = split_documents(docs, strategy=ChunkingStrategy.PAGE_LEVEL)

        # Hierarchical
        result = split_documents(docs, strategy="hierarchical")
        child_chunks = result["child"]    # → masuk vector store
        parent_chunks = result["parent"]  # → simpan di docstore

        # Semantic
        chunks = split_documents(docs, strategy="semantic",
                                 breakpoint_threshold_type="percentile",
                                 breakpoint_threshold_amount=95.0)

        # Agentic
        chunks = split_documents(docs, strategy="agentic")
    """
    strategy = ChunkingStrategy(strategy)

    dispatch = {
        ChunkingStrategy.RECURSIVE: chunk_recursive,
        ChunkingStrategy.DOCUMENT_AWARE: chunk_document_aware,
        ChunkingStrategy.PAGE_LEVEL: chunk_page_level,
        ChunkingStrategy.HIERARCHICAL: chunk_hierarchical,
        ChunkingStrategy.SEMANTIC: chunk_semantic,
        ChunkingStrategy.AGENTIC: chunk_agentic,
    }

    fn = dispatch[strategy]
    return fn(documents, **kwargs)
