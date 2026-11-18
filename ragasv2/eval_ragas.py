"""
RAGAS Evaluation Module untuk Virtual Muthawwif Cerdas
Mengevaluasi kualitas RAG chatbot menggunakan metrik RAGAS:
- Faithfulness: Apakah jawaban sesuai dengan konteks (tidak halusinasi)
- Answer Relevancy: Relevansi jawaban terhadap pertanyaan
- Context Precision: Apakah dokumen relevan muncul di ranking atas
- Context Recall: Apakah semua informasi penting ter-retrieve
"""

import json
import sys
import time
from pathlib import Path

# Allow importing from the project root when run from ragasv2/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Judge LLM untuk seluruh metrik RAGAS. Jangan ganti tanpa menjalankan ulang
# semua run yang akan dibandingkan — skor antar judge tidak sebanding.
JUDGE_MODEL = "deepseek-v4-pro"

# Gaya prompt penjawab saat evaluasi. "ketat" membuang sapaan, doa, dan tawaran
# pertanyaan lanjutan yang tidak mungkin ditelusuri ke konteks dan karenanya
# menekan faithfulness. Produksi (app.py) tetap memakai "ramah".
GAYA_EVALUASI = "ketat"

# Perluasan query sebelum retrieval: menjembatani bahasa awam jemaah dengan
# istilah fiqih di korpus. "mati" | "glosarium" | "llm" | "gabung".
#
# Sengaja "mati" pada run ground truth revisi. Uji plafon menunjukkan perluasan
# JUSTRU menurunkan cakupan pada retrieval dense (0,447 -> 0,426): menambah
# belasan istilah menggeser titik embedding menjauh dari maksud asli. Lagipula
# retrieval harus identik dengan baseline agar perubahan metrik konteks murni
# mencerminkan ground truth baru, bukan campuran beberapa sebab.
MODE_PERLUASAN = "mati"
from langchain_community.embeddings import HuggingFaceEmbeddings
from vector_manager import get_vector_store
from config import DOCS_DIR, CHAT_ENGINE, EMBEDDING_MODEL, RETRIEVE_K, RERANK_K

# Konfigurasi embeddings untuk RAGAS (gunakan model lokal untuk menghindari API limit)
_hf_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
embeddings = LangchainEmbeddingsWrapper(_hf_embeddings)
from openai import OpenAI
# Import dependencies berdasarkan CHAT_ENGINE default
if CHAT_ENGINE == "openai":
    print("[*] Menggunakan Chat Engine: OpenAI (chat_engine_v2)")
    from chat_engine_v2 import get_chat_client, get_reranker, rerank_documents, generate_response
    from langchain_openai import ChatOpenAI
    #from langchain_google_genai import ChatGoogleGenerativeAI
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, GEMINI_API_KEY, GEMINI_MODEL
# else:
#     print("[*] Menggunakan Chat Engine: Gemini (chat_engine)")
#     from chat_engine import get_chat_client, get_reranker, rerank_documents, generate_response
#     from langchain_openai import ChatOpenAI
#     #from langchain_google_genai import ChatGoogleGenerativeAI
#     from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, GEMINI_API_KEY, GEMINI_MODEL

def get_ragas_llm(model_name=None, model_type=None):
    """
    Membuat instance LLM untuk RAGAS berdasarkan model yang dipilih.
    
    Returns LangchainLLMWrapper yang dibutuhkan ragas 0.4.x.
    """
    if not model_name or not model_type:
        if CHAT_ENGINE == "openai":
            print(f"[*] RAGAS LLM: OpenAI {OPENAI_MODEL} (default)")
            lc_llm = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=0,
                max_tokens=4096,
                max_retries=3,
            )
        # else:
        #     print(f"[*] RAGAS LLM: Gemini {GEMINI_MODEL} (default)")
        #     lc_llm = ChatGoogleGenerativeAI(
        #         model=GEMINI_MODEL,
        #         google_api_key=GEMINI_API_KEY,
        #         temperature=0,
        #     )
        return LangchainLLMWrapper(lc_llm)

    if model_type == "openai":
        print(f"[*] RAGAS LLM: OpenAI {model_name}")
        lc_llm = ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0,
            max_tokens=4096,
            max_retries=3,
        )
    # elif model_type == "gemini":
    #     print(f"[*] RAGAS LLM: Gemini {model_name}")
    #     lc_llm = ChatGoogleGenerativeAI(
    #         model=model_name,
    #         google_api_key=GEMINI_API_KEY,
    #         temperature=0,
    #     )
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'openai' or 'gemini'")
    return LangchainLLMWrapper(lc_llm)

def load_test_dataset(filepath="test_dataset.json"):
    """
    Memuat dataset evaluasi dari file JSON.
    Format:
    {
        "questions": ["...", "..."],
        "ground_truths": [["..."], ["..."]],
        "metadata": [{"category": "rukun"}, ...]
    }
    """
    print(f"[*] Memuat dataset evaluasi dari {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"[✔] Dataset dimuat: {len(data['questions'])} pertanyaan")
    return data

def run_rag_pipeline(question, vector_store, client, reranker, model_name=None):
    """
    Menjalankan pipeline RAG lengkap: retrieve → rerank → generate.
    
    Args:
        question: Pertanyaan user
        vector_store: Vector store untuk retrieval
        client: Chat client (Gemini atau OpenAI)
        reranker: Cross-encoder untuk reranking
        top_k: Jumlah dokumen yang diretrieve
        model_name: (Opsional) Nama model untuk generate response
    
    Return: answer, contexts (list of strings)
    """
    # 1. Perluas query — hanya untuk pencarian. Pertanyaan asli tetap dipakai
    #    saat reranking dan saat generate, karena cross-encoder dan LLM harus
    #    menilai terhadap pertanyaan sebenarnya, bukan terhadap daftar istilah.
    query_cari = question
    if MODE_PERLUASAN != "mati":
        from query_expansion import perluas
        query_cari = perluas(question, mode=MODE_PERLUASAN)

    # 2. Retrieve dokumen
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_K})
    retrieved_docs = retriever.invoke(query_cari)

    # 3. Rerank dokumen — RETRIEVE_K kandidat disaring jadi RERANK_K,
    #    mengikuti pipeline app.py agar evaluasi mengukur chatbot yang sebenarnya.
    reranked = rerank_documents(question, retrieved_docs, reranker, top_k=RERANK_K)
    contexts = [doc.page_content for doc, score in reranked]

    # 4. Generate jawaban — stream=True because generate_response is a generator function
    context_text = "\n\n".join(contexts)
    answer = "".join(generate_response(client, question, context_text, stream=True,
                                       model_name=model_name, gaya=GAYA_EVALUASI))

    return answer, contexts

def save_rag_results_csv(questions, answers, all_contexts, output_dir="."):
    """Menyimpan hasil run_rag_pipeline ke CSV dengan nama rag_run_<timestamp>.csv."""
    import csv
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(output_dir) / f"rag_run_{timestamp}.csv"

    max_ctx = max((len(c) for c in all_contexts), default=0)
    header = ["question", "answer"] + [f"context_{i+1}" for i in range(max_ctx)]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q, a, ctxs in zip(questions, answers, all_contexts):
            row = [q, a] + ctxs + [""] * (max_ctx - len(ctxs))
            writer.writerow(row)

    print(f"[\u2714] Hasil RAG pipeline disimpan ke {filepath}")
    return str(filepath)

def evaluate_ragas(test_data, vector_store, client, reranker, base_model=None, model_type=None, result_label="default"):
    """
    Menjalankan evaluasi RAGAS pada dataset test.
    
    Args:
        test_data: Dataset test dengan questions dan ground_truths
        vector_store: Vector store untuk retrieval
        client: Chat client (Gemini atau OpenAI)
        reranker: Cross-encoder untuk reranking
        base_model: Nama model untuk API call LLM (e.g., "gpt-4o-mini", "gemini-2.5-flash-lite")
        model_type: Tipe engine ("openai" atau "gemini")
        result_label: Label lengkap untuk naming file hasil (e.g., "gpt-4o-mini-baseline")
    """
    questions = test_data["questions"]
    ground_truths = test_data["ground_truths"]
    
    answers = []
    all_contexts = []
    
    print("\n[*] Menjalankan RAG pipeline untuk setiap pertanyaan...")
    if base_model:
        print(f"[*] Model yang digunakan: {base_model}")
    
    for i, question in enumerate(questions):
        print(f"\n--- Pertanyaan {i+1}/{len(questions)} ---")
        print(f"Q: {question}")
        
        answer, contexts = run_rag_pipeline(question, vector_store, client, reranker, model_name=base_model)
        
        answers.append(answer)
        all_contexts.append(contexts)
        
        print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        
        if i < len(questions) - 1:
            time.sleep(2)

    save_rag_results_csv(questions, answers, all_contexts)

    # Ragas 0.4.x: pakai EvaluationDataset + SingleTurnSample
    ground_truths_flat = [gt[0] if isinstance(gt, list) else gt for gt in ground_truths]
    
    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=ref,
        )
        for q, a, c, ref in zip(questions, answers, all_contexts, ground_truths_flat)
    ]
    dataset = EvaluationDataset(samples=samples)
    
    print("\n[*] Menjalankan evaluasi RAGAS...")
    print("[*] Metrik: Context Precision, Context Recall, Context Entity Recall, Answer Relevancy, Faithfulness")
    
    # deepseek-v4-pro adalah reasoning model: reasoning_tokens ikut dihitung ke
    # max_tokens. Kalau max_tokens terlalu kecil, content balik kosong dan RAGAS
    # menghasilkan NaN — persis kegagalan answer_relevancy pada run mimo-v2.5,
    # yang menghasilkan NaN di 33 dari 33 sampel.
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
        max_tokens=16384,
        max_retries=5,
    ))

    metrics = [
        ContextPrecision(),
        ContextRecall(),
        ContextEntityRecall(),
        # strictness=1: default (3) meminta 3 completion sekaligus per call,
        # yang ditolak (400 Param Incorrect) oleh gateway sumopod.
        AnswerRelevancy(strictness=1),
        Faithfulness(),
    ]

    # Concurrency rendah + retry banyak + timeout longgar: reasoning model jauh
    # lebih lambat per call, dan gateway sumopod pernah gagal sebagian.
    run_config = RunConfig(max_workers=3, max_retries=15, max_wait=90, timeout=600)

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=embeddings,
        experiment_name=result_label,
        run_config=run_config,
    )

    return result

def print_evaluation_results(result):
    """
    Menampilkan hasil evaluasi dengan format yang mudah dibaca.
    """
    print("\n" + "="*60)
    print("📊 HASIL EVALUASI RAGAS")
    print("="*60)
    
    metrics = result.to_pandas()

    # Kolom yang mungkin dihasilkan oleh masing-masing metric class
    metric_cols = [       
        ("Context Precision",    "context_precision"),
        ("Context Recall",       "context_recall"),
        ("Context Entity Recall","context_entity_recall"),
        ("Answer Relevancy",     "answer_relevancy"),
        ("Faithfulness",         "faithfulness"),
    ]
    present = [(label, col) for label, col in metric_cols if col in metrics.columns]

    print("\n📈 Rata-rata Skor:")
    for label, col in present:
        val = metrics[col].mean()
        print(f"  • {label:<25}: {val:.4f}")

    print("\n📋 Detail per Pertanyaan:")
    print(metrics[[col for _, col in present]])

    print("\n💡 Interpretasi:")
    scores = {col: metrics[col].mean() for _, col in present}

    if "context_precision" in scores:
        v = scores["context_precision"]
        if v >= 0.8:   print("  ✅ Context Precision tinggi: Dokumen relevan di ranking atas")
        elif v >= 0.6: print("  ⚠️  Context Precision sedang: Beberapa dokumen tidak relevan")
        else:          print("  ❌ Context Precision rendah: Retrieval kurang presisi")

    if "context_recall" in scores:
        v = scores["context_recall"]
        if v >= 0.8:   print("  ✅ Context Recall tinggi: Semua info penting ter-retrieve")
        elif v >= 0.6: print("  ⚠️  Context Recall sedang: Beberapa info penting terlewat")
        else:          print("  ❌ Context Recall rendah: Banyak info penting tidak ditemukan")

    if "context_entity_recall" in scores:
        v = scores["context_entity_recall"]
        if v >= 0.8:   print("  ✅ Context Entity Recall tinggi: Entitas penting ter-retrieve dengan baik")
        elif v >= 0.6: print("  ⚠️  Context Entity Recall sedang: Beberapa entitas penting terlewat")
        else:          print("  ❌ Context Entity Recall rendah: Banyak entitas penting tidak ditemukan")

    if "answer_relevancy" in scores:
        v = scores["answer_relevancy"]
        if v >= 0.8:   print("  ✅ Answer Relevancy tinggi: Jawaban sangat relevan")
        elif v >= 0.6: print("  ⚠️  Answer Relevancy sedang: Beberapa jawaban kurang fokus")
        else:          print("  ❌ Answer Relevancy rendah: Jawaban sering melenceng")    

    if "faithfulness" in scores:
        v = scores["faithfulness"]
        if v >= 0.8:   print("  ✅ Faithfulness tinggi: Bot jarang halusinasi")
        elif v >= 0.6: print("  ⚠️  Faithfulness sedang: Ada jawaban tidak sesuai konteks")
        else:          print("  ❌ Faithfulness rendah: Banyak halusinasi, perlu perbaikan prompt")    

    print("\n" + "="*60)

def save_results(result, output_file="ragas_results.json", model_name=None):
    """
    Menyimpan hasil evaluasi ke file JSON.
    Args:
        output_file: Nama file output (akan diabaikan jika model_name ada)
        model_name: Nama model untuk auto-generate filename
    """
    # Auto-generate filename jika model_name diberikan
    if model_name:
        output_file = f"ragas_results_{model_name}.json"
    import numpy as np
    
    # Convert pandas DataFrame to dict
    results_dict = result.to_pandas().to_dict(orient='records')
    
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Convert all values in results_dict
    results_dict = convert_to_serializable(results_dict)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n[✔] Hasil evaluasi disimpan ke {output_file}")

if __name__ == "__main__":
    # Inisialisasi komponen
    print("[*] Inisialisasi komponen RAG...")
    vector_store = get_vector_store()
    client = get_chat_client()
    reranker = get_reranker()
    
    # Load dataset
    test_data = load_test_dataset("test_dataset.json")
    
    # Jalankan evaluasi dengan model default dari config
    result = evaluate_ragas(
        test_data, 
        vector_store, 
        client, 
        reranker,
        base_model=None,      # None = gunakan default dari config
        model_type=None       # None = gunakan default dari config
    )
    
    # Tampilkan dan simpan hasil
    print_evaluation_results(result)
    save_results(result)