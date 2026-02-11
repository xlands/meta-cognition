#!/bin/bash
# ============================================================================
# Experiment 1: Generate Prompt Pairs for 6 Metacognitive Dimensions
# ============================================================================
#
# Dimensions: evaluation_awareness, self_assessed_capability, perceived_risk,
#             computational_effort, audience_expertise, intentionality
#
# Data sources: GSM8K (dims 1-4), MMLU-Pro (dims 5-6)
# ============================================================================

set -e

NUM_SAMPLES=${1:-200}
SEED=${2:-42}
OUTPUT_DIR="data/prompts"

echo "=========================================="
echo "Experiment 1: Generate Prompt Pairs"
echo "  Samples per dimension: ${NUM_SAMPLES}"
echo "  Seed: ${SEED}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

python3 -m src.experiments.01_generate_prompts \
    --dimensions all \
    --num_samples ${NUM_SAMPLES} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "Done! Check ${OUTPUT_DIR}/dim_*.json"
