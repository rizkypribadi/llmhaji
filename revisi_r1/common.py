"""Komponen bersama untuk eksperimen Revisi R1.

Prinsip: konfigurasi IDENTIK dengan run final (ragasv2/run_eval_v4.py +
ragasv2/eval_ragas.py) kecuali satu faktor yang diuji. Semua nilai di bawah
disalin dari kode final; jangan diubah tanpa mencatatnya di config_snapshot.
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ragasv2"))

RESULTS = ROOT / "revisi_r1" / "results"
DATASET_FILE = ROOT / "ragasv2" / "pertanyaan_dataset_gabungan.json"
BASELINE_JSON = ROOT / "ragasv2" / "ragas_results_v4_deepseek.json"

METRICS = ["context_precision", "context_recall", "context_entity_recall",
           "answer_relevancy", "faithfulness"]

# --- Salinan persis dari kode final -------------------------------------------
JUDGE_MODEL = "deepseek-v4-pro"           # eval_ragas.py:33
# Revisi R1 (keputusan peneliti, 14 Nov 2026, demi biaya): semua penilaian yang BELUM selesai
# memakai judge ini. Satu eksperimen = satu judge; generator tetap deepseek-v4-pro.
R1_JUDGE = "gemini/gemini-3.1-flash-lite"
JUDGE_TEMPERATURE = 0                      # eval_ragas.py:236
JUDGE_MAX_TOKENS = 16384                   # eval_ragas.py:237
JUDGE_MAX_RETRIES = 5                      # eval_ragas.py:238
RUNCFG = dict(max_workers=3, max_retries=15, max_wait=90, timeout=600)  # eval_ragas.py:256
ANSWER_STRICTNESS = 1                      # eval_ragas.py:246

ENFORCED_FALLBACK = ("Informasi tersebut tidak dapat dipastikan dari rujukan; "
                     "silakan konsultasikan dengan pembimbing ibadah.")


def load_dataset():
    d = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    refs = [g[0] if isinstance(g, list) else g for g in d["ground_truths"]]
    return d["questions"], refs, d["metadata"]


def is_nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def _judge_client_kwargs(provider: str):
    """Kredensial judge. 'sumopod' = gateway yang dipakai run final."""
    if provider == "sumopod":
        from config import OPENAI_API_KEY, OPENAI_BASE_URL
        return dict(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if provider == "sumopod_r1":  # key sumopod kedua (revisi_r1/LLM API Key.txt -> .env)
        return dict(api_key=os.environ["SUMOPOD_R1_API_KEY"], base_url=os.environ["SUMOPOD_R1_BASE_URL"])
    if provider == "openrouter":
        return dict(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    if provider == "groq":
        return dict(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    if provider == "openai":
        return dict(api_key=os.environ["OPENAI_DIRECT_API_KEY"], base_url="https://api.openai.com/v1")
    raise ValueError(provider)


# Semua eksperimen R1 memakai key sumopod kedua (keputusan pengguna 13 Nov 2026).
# Model & parameter tetap sama dengan run final; hanya akun penagihan berbeda.
DEFAULT_PROVIDER = "sumopod_r1"


def build_chat_client(provider=DEFAULT_PROVIDER):
    """Client generator (OpenAI-compatible), pengganti chat_engine_v2.get_chat_client()."""
    from openai import OpenAI
    return OpenAI(**_judge_client_kwargs(provider))


TOKENS = {"prompt": 0, "completion": 0, "calls": 0}  # akumulasi pemakaian judge (estimasi biaya)


def _counting_chat_cls():
    from langchain_openai import ChatOpenAI

    class CountingChatOpenAI(ChatOpenAI):
        def _tally(self, res):
            u = (res.llm_output or {}).get("token_usage") or {}
            TOKENS["prompt"] += u.get("prompt_tokens", 0) or 0
            TOKENS["completion"] += u.get("completion_tokens", 0) or 0
            TOKENS["calls"] += 1
            return res

        def _generate(self, *a, **k):
            return self._tally(super()._generate(*a, **k))

        async def _agenerate(self, *a, **k):
            return self._tally(await super()._agenerate(*a, **k))

    return CountingChatOpenAI


def build_judge(judge_model=JUDGE_MODEL, provider=DEFAULT_PROVIDER, max_tokens=JUDGE_MAX_TOKENS):
    from ragas.llms import LangchainLLMWrapper
    ChatOpenAI = _counting_chat_cls()
    return LangchainLLMWrapper(ChatOpenAI(
        model=judge_model, temperature=JUDGE_TEMPERATURE, max_tokens=max_tokens,
        max_retries=JUDGE_MAX_RETRIES, **_judge_client_kwargs(provider)))


_EMB = None


def build_embeddings():
    """Sama dengan eval_ragas.py:55-56 (bge-m3 lokal via langchain_community)."""
    global _EMB
    if _EMB is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from config import EMBEDDING_MODEL
        _EMB = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    return _EMB


def build_metrics(names=METRICS):
    from ragas.metrics import (Faithfulness, AnswerRelevancy, ContextPrecision,
                               ContextRecall, ContextEntityRecall)
    table = {
        "context_precision": lambda: ContextPrecision(),
        "context_recall": lambda: ContextRecall(),
        "context_entity_recall": lambda: ContextEntityRecall(),
        "answer_relevancy": lambda: AnswerRelevancy(strictness=ANSWER_STRICTNESS),
        "faithfulness": lambda: Faithfulness(),
    }
    return [table[n]() for n in names]


def ragas_score_rows(rows, cache_path: Path, judge=None, metric_names=METRICS,
                     chunk=10, label="revisi_r1", judge_workers=None, log=print, nan_retries=2):
    """Nilai rows (dict: user_input, retrieved_contexts, response, reference) dengan RAGAS.

    Resume: skor per pertanyaan di-cache ke cache_path (JSONL, kunci 'idx').
    Dievaluasi per potongan `chunk` soal agar run yang terputus tidak hilang semua.
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample
    from ragas.run_config import RunConfig

    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["idx"]] = r
    if judge is None:
        judge = build_judge()
    emb = build_embeddings()
    cfg = dict(RUNCFG)
    if judge_workers:
        cfg["max_workers"] = judge_workers
    todo = [i for i in range(len(rows)) if i not in done]
    log(f"[ragas] {len(done)} di cache, {len(todo)} perlu dinilai")
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        samples = [SingleTurnSample(user_input=rows[i]["user_input"], response=rows[i]["response"] or "",
                                    retrieved_contexts=rows[i]["retrieved_contexts"],
                                    reference=rows[i]["reference"]) for i in part]
        res = evaluate(EvaluationDataset(samples=samples), metrics=build_metrics(metric_names),
                       llm=judge, embeddings=emb, experiment_name=label,
                       run_config=RunConfig(**cfg), show_progress=False)
        df = res.to_pandas()
        with cache_path.open("a", encoding="utf-8") as f:
            for i, (_, rec) in zip(part, df.iterrows()):
                out = {"idx": i}
                for m in metric_names:
                    v = rec.get(m)
                    out[m] = None if is_nan(v) else float(v)
                done[i] = out
                f.write(json.dumps(out) + "\n")
        log(f"[ragas] {len(done)}/{len(rows)} selesai")

    # Tambal NaN — padanan ragasv2/tambal_nan.py pada run final: hanya sel NaN yang
    # dinilai ulang (penyebab umumnya LLMDidNotFinishException/Timeout), sel lain tidak
    # disentuh. Jawaban kosong dilewati karena faithfulness-nya memang tak terdefinisi.
    for attempt in range(1, nan_retries + 1):
        for m in metric_names:
            idxs = [i for i in range(len(rows)) if done[i].get(m) is None
                    and not (m == "faithfulness" and not (rows[i]["response"] or "").strip())]
            if not idxs:
                continue
            log(f"[tambal-nan #{attempt}] {m}: {len(idxs)} sel")
            samples = [SingleTurnSample(user_input=rows[i]["user_input"], response=rows[i]["response"] or "",
                                        retrieved_contexts=rows[i]["retrieved_contexts"],
                                        reference=rows[i]["reference"]) for i in idxs]
            res = evaluate(EvaluationDataset(samples=samples), metrics=build_metrics([m]), llm=judge,
                           embeddings=emb, experiment_name=f"{label}_nan", run_config=RunConfig(**cfg),
                           show_progress=False)
            with cache_path.open("a", encoding="utf-8") as f:
                for i, (_, rec) in zip(idxs, res.to_pandas().iterrows()):
                    v = rec.get(m)
                    if not is_nan(v):
                        done[i][m] = float(v)
                        done[i].setdefault("patched", []).append(m)
                    f.write(json.dumps(done[i]) + "\n")
    return [done[i] for i in range(len(rows))]
