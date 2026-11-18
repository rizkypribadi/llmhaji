from openai import OpenAI
from sentence_transformers import CrossEncoder
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, RERANKER_MODEL
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Anggaran token untuk satu jawaban. Pada reasoning model (deepseek-v4-pro),
# reasoning_tokens dipotong dari anggaran yang sama, jadi nilai ini harus jauh
# di atas panjang jawaban yang diharapkan.
ANSWER_MAX_TOKENS = 8192


def get_chat_client():
    """
    Inisialisasi OpenAI-compatible API Client.
    """
    print("[*] Inisialisasi OpenAI API Client...")
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    return client


def get_reranker():
    """
    Inisialisasi Cross-Encoder untuk Reranking hasil pencarian.
    """
    print(f"[*] Inisialisasi Reranker: {RERANKER_MODEL}...")
    reranker = CrossEncoder(RERANKER_MODEL)
    return reranker


def rerank_documents(query: str, documents: list, reranker: CrossEncoder, top_k: int = 3) -> list:
    """
    Melakukan reranking pada dokumen yang telah di-retrieve untuk akurasi lebih tinggi.
    """
    if not documents:
        return []

    print("[*] Melakukan reranking dokumen...")
    # Buat pasangan (Query, Dokumen) untuk dievaluasi oleh Cross-Encoder
    pairs = [(query, doc.page_content) for doc in documents]

    # Hitung skor relevansi
    scores = reranker.predict(pairs)

    # Gabungkan dokumen dengan skornya, lalu urutkan dari skor tertinggi
    # reranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)   

    # Tampilkan hasil pemeringkatan ulang
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Hasil setelah Reranking:")
    for idx, (doc, score) in enumerate(scored_docs, 1):
        print(f"{idx}. Skor: {score:.4f} | Sumber: {doc.metadata.get('source', 'Unknown')} | Konten: {doc.page_content[:100]}...")

    #return [doc for doc, score in reranked_results[:top_k]]  # Ambil top_k terbaik 

    # Ambil top_k terbaik saja
    return scored_docs[:top_k]


PROMPT_RAMAH = """Anda adalah Virtual Muthawwif Cerdas, pemandu haji & umrah profesional sesuai sunnah.

        KEAHLIAN & CAKUPAN:
        - Fiqh & tata cara manasik (ihram, thawaf, sa'i, wukuf, mabit, jumrah, tahallul, dam/fidyah).
        - Doa, dzikir, ziarah tempat bersejarah, dan tips praktis di Makkah & Madinah.
        - Perbandingan madzhab (Syafi'i, Hanafi, Maliki, Hanbali) secara adil dengan dalil Al-Quran/Hadits sahih.

        ATURAN UTAMA:
        - Jawab dengan bahasa Indonesia yang santun, sabar, penuh hikmah, jelas, dan ringkas.
        - Pahami typo jamaah (misal: "twaf" -> thawaf). Jika pesan ambigu/singkat (misal: "dam"), jelaskan umum dan tawarkan pertanyaan spesifik.
        - Sapaan/pesan iseng cukup dibalas ramah sambil menawarkan bantuan seputar haji/umrah.
        - DILARANG mengarang/membuat dalil palsu. Jika info tidak ada di konteks/data, wajib jawab: "Maaf 🙏, saya belum menemukan informasi tersebut di data saya. Silakan konsultasikan dengan ustadz/pembimbing haji Anda untuk memperoleh kepastian."

        FORMAT JAWABAN:
        - Gunakan HANYA format teks biasa dan **bold**. DILARANG menggunakan heading (##), tabel (|), atau blok kode (``).
        - Struktur: 1) Jawaban utama + dalil relevan, 2) Penjelasan singkat/hikmah (opsional), 3) Saran pertanyaan lanjutan terkait ibadah (opsional).

        FORMAT JAWABAN:
        - Jawaban utama (dengan dalil jika ada)
        - (Opsional) Penjelasan singkat atau hikmah
        - (Opsional) Pertanyaan lanjutan terkait ibadah
        """

# Prompt untuk EVALUASI, bukan untuk jemaah.
#
# Metrik faithfulness memverifikasi SETIAP klaim dalam jawaban terhadap konteks.
# Sapaan, doa penutup, dan tawaran pertanyaan lanjutan mustahil ada di kitab
# manasik, sehingga semuanya terhitung klaim tak terdukung. Pada run mutawwif_v2,
# 17 dari 33 jawaban memuat embel-embel semacam itu, dan sepertiga jawaban
# terpanjang ber-faithfulness 0,457 versus 0,614 pada sepertiga terpendek.
#
# Prompt ini membuang semua yang tidak berasal dari konteks. Konsekuensinya
# jawaban terasa kaku — itu memang disengaja, dan sebabnya PROMPT_RAMAH tetap
# dipertahankan sebagai default untuk produksi.
PROMPT_KETAT = """Anda adalah sistem penjawab pertanyaan fiqih haji dan umrah.

        ATURAN MUTLAK:
        - Jawab HANYA berdasarkan KONTEKS yang diberikan. Dilarang menambahkan
          pengetahuan dari luar konteks, termasuk dalil, nama ulama, atau
          pendapat madzhab yang tidak tertulis di konteks.
        - Jika konteks tidak memuat jawabannya, tulis persis: "Informasi tersebut
          tidak ditemukan dalam konteks yang tersedia." Jangan menambahkan apa pun.
        - DILARANG menulis salam, sapaan, pujian, doa, harapan, atau penutup.
        - DILARANG menawarkan pertanyaan lanjutan.
        - DILARANG menambahkan hikmah, motivasi, atau nasihat yang tidak ada di konteks.

        GAYA:
        - Langsung ke jawaban pada kalimat pertama. Tanpa pembuka.
        - Ringkas dan padat. Setiap kalimat harus dapat ditelusuri ke konteks.
        - Bahasa Indonesia baku. Teks biasa dan **bold** saja.
        """

GAYA_PROMPT = {"ramah": PROMPT_RAMAH, "ketat": PROMPT_KETAT}


def generate_response(client: OpenAI, query: str, context: str, stream: bool = True,
                      model_name: str = None, gaya: str = "ramah"):
    """
    Meminta LLM untuk memberikan jawaban berdasarkan konteks.
    Mendukung streaming maupun non-streaming.

    Args:
        client: OpenAI API client
        query: Pertanyaan user
        context: Konteks dari dokumen yang diretrieve
        stream: Jika True → men-yield setiap token teks. Jika False → return teks lengkap sebagai string
        model_name: (Opsional) Override model name. Jika None, gunakan OPENAI_MODEL dari config
        gaya: "ramah" (default, untuk produksi) atau "ketat" (untuk evaluasi RAGAS)
    """
    print("[*] Bot sedang berpikir...")

    # Gunakan model_name yang diberikan atau fallback ke config
    model_to_use = model_name if model_name else OPENAI_MODEL
    print(f"[*] Menggunakan model: {model_to_use} (gaya: {gaya})")

    if gaya not in GAYA_PROMPT:
        raise ValueError(f"gaya tidak dikenal: {gaya}. Pilih 'ramah' atau 'ketat'.")
    system_prompt = GAYA_PROMPT[gaya]

    user_message = f"""KONTEKS:
                    {context}

                    PERTANYAAN JAMAAH:
                    {query}

                    JAWABAN:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # ANSWER_MAX_TOKENS besar karena deepseek-v4-pro adalah reasoning model:
    # reasoning_tokens ikut dihitung ke max_tokens. Dengan 1024, seluruh anggaran
    # habis dipakai berpikir dan content balik KOSONG — jawaban hilang tanpa error.
    if stream:
        response_stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0,
            top_p=0.1,
            stream=True,
        )
        for chunk in response_stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta
    else:
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0,
            top_p=0.1,
            stream=False,
        )
        return response.choices[0].message.content


def generate_answer(client: OpenAI, query: str, context: str, stream: bool = True, model_name: str = None):
    """
    Meminta LLM untuk memberikan jawaban berdasarkan konteks.
    Mendukung streaming maupun non-streaming.

    Args:
        client: OpenAI API client
        query: Pertanyaan user
        context: Konteks dari dokumen yang diretrieve
        stream: Jika True → men-yield setiap token teks. Jika False → return teks lengkap sebagai string
        model_name: (Opsional) Override model name. Jika None, gunakan OPENAI_MODEL dari config
    """
    print("[*] Bot sedang berpikir...")
    
    # Gunakan model_name yang diberikan atau fallback ke config
    model_to_use = model_name if model_name else OPENAI_MODEL
    print(f"[*] Menggunakan model: {model_to_use}")

    llm = ChatOpenAI(
        model=model_to_use, # Recommended default model for RAG tasks
        temperature=0.0,     # Lower temperature ensures more accurate fact-retrieval responses
    )

    system_prompt_text = """Anda adalah Virtual Muthawwif Cerdas, pemandu haji & umrah profesional sesuai sunnah.

        KEAHLIAN & CAKUPAN:
        - Fiqh & tata cara manasik (ihram, thawaf, sa'i, wukuf, mabit, jumrah, tahallul, dam/fidyah).
        - Doa, dzikir, ziarah tempat bersejarah, dan tips praktis di Makkah & Madinah.
        - Perbandingan madzhab (Syafi'i, Hanafi, Maliki, Hanbali) secara adil dengan dalil Al-Quran/Hadits sahih.

        ATURAN UTAMA:
        - Jawab dengan bahasa Indonesia yang santun, sabar, penuh hikmah, jelas, dan ringkas.
        - Pahami typo jamaah (misal: "twaf" -> thawaf). Jika pesan ambigu/singkat (misal: "dam"), jelaskan umum dan tawarkan pertanyaan spesifik.
        - Sapaan/pesan iseng cukup dibalas ramah sambil menawarkan bantuan seputar haji/umrah.
        - DILARANG mengarang/membuat dalil palsu. Jika info tidak ada di konteks/data, wajib jawab: "Maaf 🙏, saya belum menemukan informasi tersebut di data saya. Silakan konsultasikan dengan ustadz/pembimbing haji Anda untuk memperoleh kepastian."

        FORMAT JAWABAN:
        - Gunakan HANYA format teks biasa, *bold*, dan *italic*. DILARANG menggunakan heading (##), tabel (|), atau blok kode (``).
        - Struktur: 1) Jawaban utama + dalil relevan, 2) Penjelasan singkat/hikmah (opsional), 3) Saran pertanyaan lanjutan terkait ibadah (opsional).

        FORMAT JAWABAN:
        - Jawaban utama (dengan dalil jika ada)
        - (Opsional) Penjelasan singkat atau hikmah
        - (Opsional) Pertanyaan lanjutan terkait ibadah
        

        KONTEKS:
        {context}

        PERTANYAAN:
        {input}

        JAWABAN:"""

    prompt = PromptTemplate.from_template(
        system_prompt_text
    )

    chain = prompt | llm | StrOutputParser()
    if stream:
        # Men-yield token demi token
        return chain.stream({
            "context": context,
            "input": query
        })
    else:
        # Mengembalikan string penuh sekaligus
        response = chain.invoke({
            "context": context,
            "input": query
        })
        print(f"[*] Response Generated")
        return response

    # question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    # rag_chain = create_retrieval_chain(context, question_answer_chain)
    # resp_dict = rag_chain.invoke({"input": query})   
    # print(f"[*] RAG Chain Response: {resp_dict}")
    # return resp_dict["answer"]