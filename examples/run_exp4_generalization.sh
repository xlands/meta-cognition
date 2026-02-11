#!/bin/bash
# ============================================================================
# Experiment 4: Cross-Task Generalization
# ============================================================================
#
# Apply probes trained on GSM8K / MMLU-Pro to an unseen test set (SimpleQA)
# to verify that probe directions capture task-general metacognitive states
# rather than task-specific shortcuts.
# ============================================================================

set -e

MODEL_NAME=${1:-"Qwen/Qwen3-30B-A3B"}
PROBES_DIR="data/probes"
OUTPUT_DIR="results/exp4_generalization"

MAX_SAMPLES=100
BATCH_SIZE=8
MAX_LENGTH=512
SEED=42

echo "=========================================="
echo "Experiment 4: Cross-Task Generalization"
echo "  Model: ${MODEL_NAME}"
echo "  Probes: ${PROBES_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

python3 -m src.experiments.04_generalization \
    --model_name ${MODEL_NAME} \
    --probes_dir ${PROBES_DIR} \
    --dimension all \
    --max_samples ${MAX_SAMPLES} \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --use_chat_template \
    --output_dir ${OUTPUT_DIR}

# For joint multi-dimensional steering, add:
#   --run_joint_steering --joint_dimensions all --joint_global_alphas "0.0,1.0" --normalize_vector

echo ""
echo "Done! Check ${OUTPUT_DIR}/generalization_summary_*.json"
