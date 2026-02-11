#!/bin/bash
# ============================================================================
# Experiment 3: Activation Steering
# ============================================================================
#
# Use probe directions from Experiment 2 for activation steering:
#   h' = h + alpha * w
#   alpha = { -1, 0, 1 }
#
# Evaluate steering effects on GSM8K test set.
# ============================================================================

set -e

MODEL_NAME=${1:-"Qwen/Qwen3-30B-A3B"}
PROBES_DIR="data/probes"
OUTPUT_DIR="results/exp3_steering"

ALPHAS="-1.0,0.0,1.0"
MAX_SAMPLES=200
BATCH_SIZE=4
MAX_NEW_TOKENS=256
SEED=42

echo "=========================================="
echo "Experiment 3: Activation Steering"
echo "  Model: ${MODEL_NAME}"
echo "  Probes: ${PROBES_DIR}"
echo "  Alphas: ${ALPHAS}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

python3 -m src.experiments.03_steer_activations \
    --model_name ${MODEL_NAME} \
    --probes_dir ${PROBES_DIR} \
    --dimension all \
    --alphas "${ALPHAS}" \
    --normalize_vector \
    --target_position last \
    --dataset_name openai/gsm8k \
    --dataset_config main \
    --split test \
    --prompt_field question \
    --answer_field answer \
    --max_samples ${MAX_SAMPLES} \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --use_chat_template \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "Done! Check ${OUTPUT_DIR}/steer_*.json"
