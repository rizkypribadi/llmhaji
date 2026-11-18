"""
Re-score the existing RAG answers with RAGAS (Major issues 1 and 3).
- Repeats the RAGAS judgement k times on the SAME answers -> mean ± SD across runs.
- Can use an independent judge (e.g. gpt-4o) via any OpenAI-compatible endpoint.
- Also reports per-question SD and a percentile bootstrap 95% CI.

Usage examples:
  pip install "ragas>=0.2" langchain-openai pandas numpy
  # repeated runs with the original judge (deepseek-v4-pro through your gateway)
  python rejudge_ragas.py --input ragas_results_v4_deepseek.json --judge deepseek-v4-pro \
      --base-url https://YOUR-GATEWAY/v1 --api-key $KEY --runs 5 --out runs_deepseek
  # independent judge, faithfulness only
  python rejudge_ragas.py --input ragas_results_v4_deepseek.json --judge gpt-4o \
      --api-key $OPENAI_API_KEY --runs 3 --metrics faithfulness --out runs_gpt4o
"""
import argparse, json, os, numpy as np, pandas as pd
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (LLMContextPrecisionWithReference, LLMContextRecall,
                           ContextEntityRecall, ResponseRelevancy, Faithfulness)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ap = argparse.ArgumentParser()
ap.add_argument('--input', required=True)
ap.add_argument('--judge', required=True)
ap.add_argument('--base-url', default=None)
ap.add_argument('--api-key', default=os.getenv('OPENAI_API_KEY'))
ap.add_argument('--embed-model', default='text-embedding-3-small', help='only needed for answer_relevancy')
ap.add_argument('--runs', type=int, default=3)
ap.add_argument('--temperature', type=float, default=0.0)
ap.add_argument('--max-tokens', type=int, default=4096)
ap.add_argument('--metrics', default='context_precision,context_recall,context_entity_recall,answer_relevancy,faithfulness')
ap.add_argument('--out', default='rejudge_out')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = json.load(open(a.input, encoding='utf-8'))
samples = [dict(user_input=r['user_input'], retrieved_contexts=r['retrieved_contexts'],
                response=r['response'] or '', reference=r['reference']) for r in rows]
ds = EvaluationDataset.from_list(samples)

llm = LangchainLLMWrapper(ChatOpenAI(model=a.judge, base_url=a.base_url, api_key=a.api_key,
                                     temperature=a.temperature, max_tokens=a.max_tokens))
emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=a.embed_model, base_url=a.base_url, api_key=a.api_key))
ALL = {'context_precision': LLMContextPrecisionWithReference(), 'context_recall': LLMContextRecall(),
       'context_entity_recall': ContextEntityRecall(), 'answer_relevancy': ResponseRelevancy(),
       'faithfulness': Faithfulness()}
names = a.metrics.split(',')
metrics = [ALL[n] for n in names]

per_run = []
for k in range(a.runs):
    res = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb,
                   run_config=RunConfig(max_workers=4, timeout=300, seed=42 + k))
    df = res.to_pandas(); df.to_csv(f'{a.out}/run{k+1}.csv', index=False)
    cols = [c for c in df.columns if c in names or c.replace('llm_', '') in names or c in
            ('llm_context_precision_with_reference', 'context_recall', 'answer_relevancy')]
    per_run.append(df)
    print(f'run {k+1}:', df.select_dtypes('number').mean().round(4).to_dict())

rng = np.random.default_rng(42); summary = []
for c in per_run[0].select_dtypes('number').columns:
    run_means = [df[c].mean() for df in per_run]
    v = per_run[0][c].dropna().to_numpy()
    bs = [rng.choice(v, len(v)).mean() for _ in range(10000)]
    summary.append(dict(metric=c, n=len(v), mean_over_runs=np.mean(run_means),
                        sd_across_runs=np.std(run_means, ddof=1) if len(run_means) > 1 else np.nan,
                        per_question_sd_run1=v.std(ddof=1),
                        ci95_low_run1=np.percentile(bs, 2.5), ci95_high_run1=np.percentile(bs, 97.5)))
pd.DataFrame(summary).round(4).to_csv(f'{a.out}/summary.csv', index=False)
print(pd.DataFrame(summary).round(4).to_string(index=False))
