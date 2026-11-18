import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import uvicorn

# Import modul internal
from config import DOCS_DIR, CHAT_ENGINE, ENABLE_HALLUCINATION_CHECK, NLI_MODEL, EMBEDDING_MODEL, RETRIEVE_K, RERANK_K
from data_loader import load_documents, split_documents
from vector_manager import get_vector_store, add_documents_in_batches
from hallucination_checker import get_hallucination_checker

if CHAT_ENGINE == "openai":
    print("[*] Menggunakan Chat Engine: OpenAI (chat_engine_v2)")
    from chat_engine_v2 import get_chat_client, get_reranker, rerank_documents, generate_response, generate_answer
else:
    print("[*] Menggunakan Chat Engine: Gemini (chat_engine)")
    from chat_engine import get_chat_client, get_reranker, rerank_documents, generate_response, generate_answer

# Global engine references
vector_store = None
chat_client = None
reranker = None
hallucination_checker = None  # Hallucination Mitigation Layer

def index_documents(vs):
    """
    Cek apakah koleksi kosong, jika iya lakukan indexing otomatis.
    """
    from qdrant_client import QdrantClient
    from config import QDRANT_URL, QDRANT_API_KEY, CLUSTER_NAME

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    info = client.get_collection(CLUSTER_NAME)
    point_count = info.points_count

    if point_count == 0:
        print(f"[*] Koleksi kosong. Memulai indexing dokumen dari '{DOCS_DIR}'...")
        docs = load_documents(DOCS_DIR)
        if docs:
            splits = split_documents(docs)
            add_documents_in_batches(vs, splits)
            print("[✔] Indexing selesai!")
        else:
            print("[!] Tidak ada dokumen ditemukan di folder hajj_docs/.")
    else:
        print(f"[✔] Koleksi sudah berisi {point_count} vectors, skip indexing.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: inisialisasi engine & auto-index jika koleksi kosong."""
    global vector_store, chat_client, reranker, hallucination_checker

    print("="*60)
    print("   Virtual Muthawwif Cerdas — Pemandu Haji & Umrah")
    print("="*60)
    print("[*] Menyiapkan Muthawwif Engine...")

    vector_store = get_vector_store()
    chat_client = get_chat_client()
    reranker = get_reranker()
    
    # Inisialisasi Hallucination Mitigation Layer
    if ENABLE_HALLUCINATION_CHECK:
        print("[*] Mengaktifkan Hallucination Mitigation Layer...")
        hallucination_checker = get_hallucination_checker(
            nli_model=NLI_MODEL,
            embedding_model=EMBEDDING_MODEL
        )
    else:
        print("[!] Hallucination Check DISABLED - gunakan dengan hati-hati!")

    # Auto-index jika koleksi masih kosong
    index_documents(vector_store)

    print("[✔] Engine Siap! Buka http://localhost:8000")
    yield
    print("[*] Muthawwif Engine dimatikan.")

app = FastAPI(title="Virtual Muthawwif Cerdas API", lifespan=lifespan)

# Setup static files
os.makedirs("static", exist_ok=True)
os.makedirs("assets", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

# Threshold skor reranker: di bawah ini dianggap tidak cukup relevan
RERANK_SCORE_THRESHOLD = 0.3

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health_check():
    """Health check endpoint untuk WhatsApp bridge dan monitoring."""
    return {"status": "ok", "service": "muthawwif-cerdas"}

@app.get("/api/test-dataset")
async def get_test_dataset():
    """Endpoint untuk mendapatkan test dataset evaluasi."""
    import json
    try:
        with open("test_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test dataset tidak ditemukan. Jalankan setup evaluasi terlebih dahulu.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error membaca test dataset: {str(e)}")

@app.get("/api/ragas-results")
async def get_ragas_results():
    """Endpoint untuk mendapatkan hasil evaluasi RAGAS."""
    import json
    try:
        with open("ragas_results.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Hasil evaluasi belum tersedia. Jalankan 'python run_evaluation.py' terlebih dahulu.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error membaca hasil evaluasi: {str(e)}")

@app.get("/api/evaluation-results")
async def list_evaluation_results():
    """List semua file hasil evaluasi yang tersedia."""
    import glob
    import json
    from datetime import datetime
    
    files = glob.glob("ragas_results*.json")
    results = []
    
    for filepath in files:
        try:
            # Extract model name from filename
            filename = os.path.basename(filepath)
            if filename == "ragas_results.json":
                model_name = "default"
            else:
                model_name = filename.replace("ragas_results_", "").replace(".json", "")
            
            # Get file modification time
            mtime = os.path.getmtime(filepath)
            timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # Load file to get summary stats
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Calculate average scores
            total_questions = len(data)
            avg_faithfulness = sum(item.get("faithfulness", 0) for item in data) / total_questions if total_questions else 0
            avg_relevancy = sum(item.get("answer_relevancy", 0) for item in data) / total_questions if total_questions else 0
            
            results.append({
                "filename": filename,
                "model_name": model_name,
                "timestamp": timestamp,
                "total_questions": total_questions,
                "avg_faithfulness": round(avg_faithfulness, 4),
                "avg_answer_relevancy": round(avg_relevancy, 4)
            })
        except Exception as e:
            print(f"[!] Error processing {filepath}: {e}")
            continue
    
    # Sort by timestamp descending
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return results

@app.get("/api/evaluation-results/{filename}")
async def get_evaluation_result(filename: str):
    """Get hasil evaluasi spesifik berdasarkan filename."""
    import json
    
    # Security: prevent path traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    filepath = filename if filename.endswith(".json") else f"{filename}.json"
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File {filename} tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error membaca file: {str(e)}")

class EvaluationRequest(BaseModel):
    base_model: str      # Nama model asli untuk API call LLM (e.g., "gpt-4o-mini", "gemini-2.5-flash-lite")
    model_type: str      # Engine type: "gemini" or "openai"
    result_label: str    # Label lengkap untuk naming file hasil (e.g., "gpt-4o-mini-baseline")

class EvaluationResponse(BaseModel):
    status: str
    message: str
    filename: str

@app.post("/api/run-evaluation", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """Run RAGAS evaluation dengan model yang dipilih."""
    import json
    from eval_ragas import load_test_dataset, evaluate_ragas, save_results
    
    base_model = request.base_model      # Nama model asli (e.g., "gpt-4o-mini")
    model_type = request.model_type      # Engine type ("openai" atau "gemini")
    result_label = request.result_label  # Label untuk file hasil (e.g., "gpt-4o-mini-baseline")
    
    # Validasi model_type
    if model_type not in ["gemini", "openai"]:
        raise HTTPException(status_code=400, detail="model_type harus 'gemini' atau 'openai'")
    
    # Check if test dataset exists
    if not os.path.exists("test_dataset.json"):
        raise HTTPException(status_code=404, detail="test_dataset.json tidak ditemukan")
    
    try:
        # Load test dataset
        test_data = load_test_dataset("test_dataset.json")
        
        # Import appropriate chat engine
        if model_type == "openai":
            from chat_engine_v2 import get_chat_client as get_client_v2, get_reranker as get_reranker_v2, generate_answer
            client = get_client_v2()
            reranker = get_reranker_v2()
        else:  # gemini
            from chat_engine import get_chat_client as get_client_v1, get_reranker as get_reranker_v1, generate_answer
            client = get_client_v1()
            reranker = get_reranker_v1()
        
        # Run evaluation dengan base_model untuk API call
        result = evaluate_ragas(
            test_data, 
            vector_store, 
            client, 
            reranker, 
            base_model=base_model,      # Model name untuk API call
            model_type=model_type,       # Engine type untuk RAGAS LLM
            result_label=result_label   # Label untuk naming file
        )
        
        # Save results dengan result_label
        save_results(result, model_name=result_label)
        
        filename = f"ragas_results_{result_label}.json"
        
        return EvaluationResponse(
            status="success",
            message=f"Evaluasi berhasil untuk model {base_model} ({result_label})",
            filename=filename
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error menjalankan evaluasi: {str(e)}")

class TestQuestion(BaseModel):
    question: str
    ground_truth: str
    category: str
    difficulty: str

class TestDatasetUpdate(BaseModel):
    questions: List[TestQuestion]

@app.get("/api/test-dataset-full")
async def get_test_dataset_full():
    """Get test dataset dengan format lengkap untuk editing."""
    import json
    try:
        with open("test_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Transform ke format yang lebih mudah untuk frontend
        questions = data.get("questions", [])
        ground_truths = data.get("ground_truths", [])
        metadata = data.get("metadata", [])
        
        result = []
        for i in range(len(questions)):
            gt = ground_truths[i][0] if isinstance(ground_truths[i], list) else ground_truths[i]
            meta = metadata[i] if i < len(metadata) else {"category": "", "difficulty": ""}
            
            result.append({
                "id": i,
                "question": questions[i],
                "ground_truth": gt,
                "category": meta.get("category", ""),
                "difficulty": meta.get("difficulty", "")
            })
        
        return result
    except FileNotFoundError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.put("/api/test-dataset")
async def update_test_dataset(update: TestDatasetUpdate):
    """Update seluruh test dataset."""
    import json
    
    try:
        # Transform data kembali ke format original
        questions = [q.question for q in update.questions]
        ground_truths = [[q.ground_truth] for q in update.questions]
        metadata = [{"category": q.category, "difficulty": q.difficulty} for q in update.questions]
        
        data = {
            "questions": questions,
            "ground_truths": ground_truths,
            "metadata": metadata
        }
        
        # Backup existing file
        if os.path.exists("test_dataset.json"):
            import shutil
            from datetime import datetime
            backup_name = f"test_dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy("test_dataset.json", backup_name)
        
        # Save new data
        with open("test_dataset.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {"status": "success", "message": f"Dataset berhasil diupdate dengan {len(questions)} pertanyaan"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating dataset: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint utama untuk web frontend dan WhatsApp bridge."""
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Pesan tidak boleh kosong")

    # Deteksi pesan terlalu pendek (1-2 karakter non-spasi)
    if len(query) < 3:
        return ChatResponse(
            answer="Mohon maaf 🙏, pesan Anda terlalu singkat. Silakan tulis pertanyaan lebih lengkap agar saya bisa membantu dengan baik. Contoh: \"Apa saja rukun haji?\" atau \"Bagaimana tata cara thawaf?\"",
            sources=[]
        )

    try:
        # 1. Retrieval
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_K})
        initial_docs = retriever.invoke(query)

        if not initial_docs:
            return ChatResponse(
                answer="Maaf 🙏, saya tidak menemukan dokumen yang relevan untuk pertanyaan ibadah tersebut. Silakan konsultasikan dengan pembimbing haji Anda.",
                sources=[]
            )

        # 2. Reranking
        reranked_results = rerank_documents(query, initial_docs, reranker, top_k=RERANK_K)
        print(f"[*] Reranked {len(reranked_results)} dokumen, top score: {reranked_results[0][1] if reranked_results else 'N/A'}")
        
        # 3. Filter dokumen berdasarkan skor reranker
        #high_score_docs = [(doc, score) for doc, score in reranked_results if score >= RERANK_SCORE_THRESHOLD]
        all_docs_for_context = reranked_results  # tetap pakai semua untuk konteks LLM
        #print(f"[*] {len(high_score_docs)} dokumen melewati threshold skor reranker ({RERANK_SCORE_THRESHOLD})")

        # 4. Gabungkan konteks
        context = "\n\n".join([doc.page_content for doc, score in all_docs_for_context])
        print(f"[*] Menggabungkan konteks dari {len(all_docs_for_context)} dokumen, total karakter: {len(context)}")

        
        # 5. Generate jawaban
        #answer = generate_answer(chat_client, query, context, stream=False, model_name=None)  # Gunakan model default dari config
        
        # 5. Generate jawaban
        raw = generate_response(chat_client, query, context)
        # chat_engine_v2 returns a generator (streaming); collect into a string
        answer = "".join(raw) if hasattr(raw, "__iter__") and not isinstance(raw, str) else raw
        print(f"[*] Jawaban LLM diterima : {answer}")

        # 6. Hallucination Mitigation Layer - Verifikasi jawaban
        if ENABLE_HALLUCINATION_CHECK and hallucination_checker:
            print("[*] Menjalankan Hallucination Mitigation Layer...")
            verification = hallucination_checker.verify_answer(
                answer=answer,
                context=context,
                query=query,
                sources=[doc.metadata.get("source", "") for doc, _ in all_docs_for_context if doc.metadata]
            )
            
            print(f"|-- Validation Result: {'✅ VALID' if verification['is_valid'] else '❌ HALLUCINATION'}")
            print(f"|-- Confidence: {verification['confidence']:.4f}")
            print(f"|-- Details: {verification['validation_details']}")
            
            # Disable fallback for now, just log the result

            # Jika deteksi hallucination, ganti dengan fallback message
            # if not verification['is_valid']:
            #     print("[!] Hallucination detected! Replacing with safe fallback...")
            #     answer = hallucination_checker.get_safe_fallback_message()
            #     sources = []  # Hapus sumber karena jawaban diganti
            # else:

            # Jika valid, tetap ambil sources dari high_score_docs
            if all_docs_for_context:
                sources = list(set([
                    os.path.basename(doc.metadata.get("source", ""))
                    for doc, score in all_docs_for_context
                    if doc.metadata
                ]))
            else:
                sources = []

            answer += f"""
                
                **Status:** {"✅ Valid" if verification["is_valid"] else "❌ Potential Hallucination"}
                **Confidence:** {verification["confidence"]:.2%}
                **Details:** {verification["validation_details"]}
                """    
        else:
            # Tanpa hallucination check, gunakan logika lama
            if all_docs_for_context:
                sources = list(set([
                    os.path.basename(doc.metadata.get("source", ""))
                    for doc, score in all_docs_for_context
                    if doc.metadata
                ]))
            else:
                sources = []

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        print(f"[!] API Error: {e}")
        error_msg = str(e)

        if "503" in error_msg or "UNAVAILABLE" in error_msg:
            return ChatResponse(
                answer="Mohon maaf 🙏, saat ini server sedang ramai dan butuh waktu istirahat sejenak. Silakan coba lagi dalam beberapa saat ya. Insya Allah sebentar lagi bisa kembali melayani pertanyaan Anda.",
                sources=[]
            )
        elif "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return ChatResponse(
                answer="Mohon maaf 🙏, permintaan sedang sangat banyak. Silakan tunggu sebentar dan coba kembali. Jazaakallahu khairan atas kesabarannya.",
                sources=[]
            )
        elif "400" in error_msg or "INVALID" in error_msg:
            return ChatResponse(
                answer="Maaf 🙏, saya kesulitan memproses pertanyaan tersebut. Coba sampaikan dengan kalimat yang berbeda ya.",
                sources=[]
            )

        return ChatResponse(
            answer="Maaf 🙏, terjadi gangguan teknis pada sistem. Silakan coba lagi dalam beberapa saat. Jika masih bermasalah, hubungi admin.",
            sources=[]
        )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
