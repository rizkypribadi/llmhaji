#!/usr/bin/env bash
# Melanjutkan semua job Revisi R1 dari cache (answers.jsonl / ragas.jsonl).
# Prasyarat: Docker Desktop menyala dan Qdrant (qdrant_service) jalan.
# Pakai (Git Bash, dari root repo):  bash revisi_r1/lanjutkan.sh
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
R=revisi_r1/results

# Container dihentikan manual (docker stop) agar WAL ter-flush; restart:unless-stopped
# tidak akan menyalakannya sendiri, jadi nyalakan di sini.
docker start qdrant_service >/dev/null 2>&1 || echo "[!] nyalakan Docker Desktop dulu"
until curl -s -m 3 http://127.0.0.1:6333/collections >/dev/null; do echo "menunggu Qdrant..."; sleep 5; done

# Judge semua penilaian yang belum selesai: gemini/gemini-3.1-flash-lite (common.R1_JUDGE).
# Generator tetap deepseek-v4-pro. Arsip lama: E1_deepseek_pro/, E2_gemini37flash/.
J=gemini/gemini-3.1-flash-lite
# E1: rejudge.py melanjutkan run yang belum punya ragas_results.json
next=1; while [ -f $R/E1_flashlite31/run$next/ragas_results.json ]; do next=$((next+1)); done
if [ $next -le 3 ]; then  # E1 dihentikan di k = 3 (keputusan peneliti 18 Nov)
  nohup $PY -u revisi_r1/rejudge.py --name E1_flashlite31 --judge $J --runs $((4-next)) --start-run $next --judge-workers 6 >> $R/E1_flashlite31/stdout.log 2>&1 &
fi
# E2 (judge independen) = E1_flashlite31/run1: konfigurasi identik, tidak dijalankan terpisah.
[ -f $R/A0_rerun/ragas_results.json ] || \
  nohup $PY -u revisi_r1/run_variant.py --name A0_rerun --verify advisory --workers 4 --judge-workers 5 >> $R/A0_rerun/stdout.log 2>&1 &
# Maks. 2 run_variant bersamaan (~9,5 GB commit per proses): A1 menunggu A2 selesai.
[ -f $R/A1_unconstrained/ragas_results.json ] || \
  nohup bash $R/A1_unconstrained/_chain.sh >> $R/A1_unconstrained/chain.log 2>&1 &
[ -f $R/A2_norerank/ragas_results.json ] || \
  nohup $PY -u revisi_r1/run_variant.py --name A2_norerank --rerank off --verify advisory --workers 4 --judge-workers 5 >> $R/A2_norerank/stdout.log 2>&1 &
[ -f $R/A3_indara/ragas_results.json ] || \
  nohup $PY -u revisi_r1/run_variant.py --name A3_indara --collection mutawwif_v4_indara --verify advisory --workers 4 --judge-workers 5 >> $R/A3_indara/stdout.log 2>&1 &
sleep 2; echo "job berjalan:"; tasklist | grep -ci python
