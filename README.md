# llmhaji — Evidence-grounded RAG for Indonesian Hajj & Umrah QA

Code and per-question evaluation scores for the paper *Fiqh-Constrained Retrieval-Augmented Generation for Hallucination Mitigation in Indonesian Hajj Ritual Question Answering* (IJITCS, under revision).

## Pipeline

| Stage | Component |
|---|---|
| OCR (scanned books) | Tesseract 5.4.0, `ind`, 300 dpi, cleaning in `ocr_pdf.py` |
| Chunking | recursive, 1000 characters / 200 overlap (`reindex_v4.py`) |
| Embedding | `BAAI/bge-m3` (local) |
| Vector store | Qdrant (Docker) |
| Retrieval | dense top-25 → `BAAI/bge-reranker-v2-m3` → top-5 |
| Generator | `deepseek-v4-pro` via an OpenAI-compatible gateway; temperature 0.0, top_p 0.1, max_tokens 8192, no seed |
| Verification (advisory) | `hallucination_checker.py`: NLI (`nli-deberta-v3-base`), semantic overlap, fiqh keywords, citation regexes |
| Evaluation | RAGAS 0.4.3: context precision, context recall, context entity recall, answer relevancy (strictness 1), faithfulness |

## Repository layout

```
app.py, chat_engine_v2.py, vector_manager.py, data_loader.py,
chunking_strategies.py, hallucination_checker.py, query_expansion.py, ocr_pdf.py
reindex_v4.py, reindex_hybrid.py      index building
ragasv2/                              final RAGAS evaluation (run_eval_v4.py, eval_ragas.py)
revisi_r1/                            revision experiments (judge repeats, independent judge,
                                      ablations A0–A4, verification statistics, OCR CER/WER,
                                      entity search)
results/                              per-question RAGAS scores and experiment summaries
```

## Results files

All files in `results/` hold per-question scores (`no` + metrics). Question and answer text is not included.

| File | Contents |
|---|---|
| `final_v4_deepseek_scores.json` | final run, judge `deepseek-v4-pro` |
| `E1_flashlite31_run{1,2,3}_scores.json` | three re-scorings of the final answers by `gemini-3.1-flash-lite` (run1 also serves as the independent-judge comparison) |
| `A0_rerun_scores.json` | ablation A0: full system, regenerated |
| `A1_unconstrained_scores.json` | ablation A1: prompt without evidence constraints |
| `A2_norerank_scores.json` | ablation A2: no cross-encoder reranker |
| `A3_indara_scores.json` | ablation A3: scanned books re-OCRed with `ind+ara` |
| `A4_enforced_scores.json` | ablation A4: NLI verification enforced |
| `E4_final.md` | verification-layer statistics |
| `E5_ocr.md` | OCR character and word error rates |
| `entity_search.json` | corpus search for entities missed by retrieval |

## Not included

- The book corpus (`Dataset/`), which is under copyright.
- Qdrant data and snapshots.
- API keys. Copy `.env.example` to `.env` and fill it in.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env                             # fill in keys
docker run -d -p 6333:6333 --name qdrant_service qdrant/qdrant
python reindex_v4.py                             # requires your own copy of the corpus in Dataset/
python ragasv2/run_eval_v4.py
```

Tested on Python 3.12.10 and Windows 11, CPU only.
