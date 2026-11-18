import os
from dotenv import load_dotenv

load_dotenv()

# Kunci TIDAK disimpan di kode. Salin .env.example menjadi .env lalu isi.
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")          # gateway OpenAI-compatible (sumopod)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ai.sumopod.com/v1")
OPENAI_MODEL = "deepseek-v4-pro"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")         # tidak dipakai run final; diimpor eval_ragas.py
GEMINI_MODEL = "gemini-2.5-flash-lite"

CLUSTER_NAME = "mutawwif"
DOCS_DIR = "hajj_docs"
DATASET_DIR = "Dataset"          # korpus kitab TIDAK disertakan (hak cipta)

RETRIEVE_K = 25
RERANK_K = 5
TOP_K = RERANK_K
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
ENABLE_HALLUCINATION_CHECK = True
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
CHAT_ENGINE = "openai"

os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
os.environ.setdefault("OPENAI_BASE_URL", OPENAI_BASE_URL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
