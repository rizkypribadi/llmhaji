"""Tahap 0.3 (sebagian): reproduksibilitas GENERATOR saja.
Qdrant mati, jadi retrieval tidak diulang. Konteks 10 soal pertama diambil dari
ragas_results_v4_deepseek.json (retrieved_contexts), lalu jawaban dibangkitkan
ulang dengan generate_response(gaya='ketat') yang sama persis dengan run final.
"""
import json, sys, difflib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chat_engine_v2 import get_chat_client, generate_response

OUT = ROOT / "revisi_r1/results/tahap0_repro"
base = json.loads((ROOT / "ragasv2/ragas_results_v4_deepseek.json").read_text(encoding="utf-8"))
client = get_chat_client()
rows = []
for i, r in enumerate(base[:10], 1):
    ctx = "\n\n".join(r["retrieved_contexts"])
    new = "".join(generate_response(client, r["user_input"], ctx, stream=True, gaya="ketat"))
    old = r["response"] or ""
    ratio = difflib.SequenceMatcher(None, old, new).ratio()
    rows.append({"no": i, "identik": old.strip() == new.strip(), "similarity": round(ratio, 4),
                 "len_lama": len(old), "len_baru": len(new), "jawaban_lama": old, "jawaban_baru": new})
    print(f"#{i}: identik={old.strip()==new.strip()} sim={ratio:.3f} len {len(old)}->{len(new)}", flush=True)
(OUT / "repro_gen_10.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
