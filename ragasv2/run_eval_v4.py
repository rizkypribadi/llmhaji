"""
Evaluasi RAGAS penuh terhadap koleksi mutawwif_v4 (33 pertanyaan), memakai
pipeline dan judge yang sama persis dengan run_eval_full.py (yang menghasilkan
ragas_results_v2_deepseek.json) agar skornya apple-to-apple dengan mutawwif_v2.
"""

from eval_ragas import (
    load_test_dataset,
    evaluate_ragas,
    print_evaluation_results,
    save_results,
    get_chat_client,
    get_reranker,
)
from vector_manager import get_another_vector_store
from eval_ragas import JUDGE_MODEL
from config import RETRIEVE_K, RERANK_K

TARGET_COLLECTION = "mutawwif_v4"
DATASET_FILE = "pertanyaan_dataset_gabungan.json"
OUTPUT_FILE = "ragas_results_v4_deepseek.json"

if __name__ == "__main__":
    print("[*] Inisialisasi komponen RAG...")
    print(f"|-- Koleksi   : {TARGET_COLLECTION}")
    print(f"|-- Dataset   : {DATASET_FILE}")
    print(f"|-- Retrieval : {RETRIEVE_K} kandidat -> rerank {RERANK_K}")
    print(f"|-- Judge     : {JUDGE_MODEL}")

    vector_store = get_another_vector_store(collection_name=TARGET_COLLECTION)
    client = get_chat_client()
    reranker = get_reranker()

    test_data = load_test_dataset(DATASET_FILE)

    result = evaluate_ragas(
        test_data,
        vector_store,
        client,
        reranker,
        base_model=None,
        model_type=None,
        result_label="pertanyaan_dataset_v4",
    )

    print_evaluation_results(result)
    save_results(result, OUTPUT_FILE)
