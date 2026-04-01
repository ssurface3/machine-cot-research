#!/usr/bin/env bash
# ---------------------------------------------------------------
#  Machine-CoT Research -- Full Pipeline
#  Run: bash run_all.sh
# ---------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

echo "========================================================"
echo "  Phase 1: Data Synthesis (Teacher -> 5 CoT levels)"
echo "========================================================"
python phase1_data_synthesis.py

echo ""
echo "========================================================"
echo "  Phase 2: Fine-Tuning (Student x 5 levels)"
echo "========================================================"
python phase2_finetune.py

echo ""
echo "========================================================"
echo "  Phase 3: Evaluation (Test set x 5 models)"
echo "========================================================"
python phase3_evaluation.py

echo ""
echo "========================================================"
echo "  Phase 4: Analysis & Figures"
echo "========================================================"
python phase4_analysis.py

echo ""
echo "All phases complete. Check outputs/ and figures/"
