#!/bin/bash
# ============================================================================
# Experiment 2: Extract Activations, Train Probes, Analyze Orthogonality
# ============================================================================
#
# Pipeline:
#   1. Extract prompt last-token residual stream activations for each dimension
#   2. Train per-layer LogisticRegression probes
#   3. Cosine similarity + PCA analysis for orthogonality
# ============================================================================

set -e

MODEL_NAME=${1:-"Qwen/Qwen3-30B-A3B"}
PROMPTS_DIR="data/prompts"
OUTPUT_DIR="data/probes"
ACTIVATIONS_DIR="data/activations"

BATCH_SIZE=8
MAX_LENGTH=512
TEST_SIZE=0.2
SEED=42

echo "=========================================="
echo "Experiment 2: Train Probes + Orthogonality"
echo "  Model: ${MODEL_NAME}"
echo "  Prompts: ${PROMPTS_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

python3 -m src.experiments.02_train_probes \
    --model_name ${MODEL_NAME} \
    --prompts_dir ${PROMPTS_DIR} \
    --dimensions all \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --test_size ${TEST_SIZE} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --save_activations \
    --activations_dir ${ACTIVATIONS_DIR} \
    --use_chat_template

echo ""
echo "Done! Check ${OUTPUT_DIR}/probe_*.pt and orthogonality_analysis_*.json"
