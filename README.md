# Decomposing and Steering Implicit Metacognitive States in Large Language Models

This repository contains the code and data for reproducing the experiments in our paper. We study how large language models encode **implicit metacognitive states**---internal variables such as evaluation awareness, self-assessed capability, and perceived risk---that causally influence reasoning behavior.

## Key Findings

1. **Linear decodability**: Metacognitive states are linearly separable in the residual stream, with probe accuracy scaling from 0.63 (0.6B) to near-perfect 1.00 (30B+).
2. **Causal control**: Activation steering along probe directions produces dimension-specific behavioral shifts (e.g., 28% verbosity reduction, accuracy gains from 25% to 44%).
3. **Cross-task generalization**: Probes transfer to unseen domains with 81% mean accuracy, ruling out task-specific shortcuts.
4. **Joint multi-dimensional steering**: All six dimensions can be simultaneously controlled through a single superposed intervention (5/6 shift independently on the 30B model).

## Project Structure

```
.
├── config/
│   └── llm_config.example.yaml     # API config template for LLM judge
├── data/
│   ├── prompts/                     # Experiment 1: prompt pairs (6 dimensions)
│   ├── probes/                      # Experiment 2: trained probe weights + analysis
│   └── activations/                 # Experiment 2: cached residual stream activations
├── results/
│   ├── exp3_steering/               # Experiment 3: activation steering results
│   ├── exp4_generalization/         # Experiment 4: cross-task generalization results
│   └── exp4_joint/                  # Experiment 4: joint multi-dimensional steering
├── examples/                        # Shell scripts for running each experiment
├── src/
│   └── experiments/
│       ├── 01_generate_prompts.py   # Generate metacognitive prompt pairs
│       ├── 02_train_probes.py       # Extract activations, train probes, orthogonality
│       ├── 03_steer_activations.py  # Activation steering experiments
│       ├── 04_generalization.py     # Cross-task generalization + joint steering
│       └── metrics.py               # Behavioral scoring metrics
└── requirements.txt
```

## Metacognitive Dimensions

| # | Dimension | Description |
|---|-----------|-------------|
| 1 | Evaluation Awareness | Awareness of being tested or evaluated |
| 2 | Self-Assessed Capability | Confidence in ability to handle the current task |
| 3 | Perceived Risk | Perception of potential risks or sensitivity |
| 4 | Computational Effort | Allocation of reasoning depth and verbosity |
| 5 | Audience Expertise | Adaptation to the perceived expertise of the audience |
| 6 | Intentionality | Goal-directedness vs. open-ended exploration |

## Models

Experiments were conducted on five models across two architectures:

- **Qwen3-0.6B** (0.6B dense)
- **Qwen3-14B** (14B dense)
- **Qwen3-30B-A3B** (30B MoE, 3B active)
- **Qwen3-235B-A22B** (235B MoE, 22B active)
- **Llama-4-Scout-17B-16E** (109B MoE, 17B active)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Set up LLM judge API
cp config/llm_config.example.yaml config/llm_config.yaml
# Edit config/llm_config.yaml with your API key
```

## Pre-included Data

This repository ships with pre-computed artifacts so that you can **skip the expensive training steps** and jump straight to steering or generalization experiments.

| Directory | Contents | Size (approx.) |
|-----------|----------|-----------------|
| `data/prompts/` | Binary-contrast prompt pairs for all 6 dimensions (GSM8K + MMLU-Pro) | ~50 MB |
| `data/activations/` | Cached residual-stream activations for all 5 models x 6 dimensions | ~2 GB (Git LFS) |
| `data/probes/` | Trained probe weights (`.pt`) + per-layer accuracy summaries (`.json`) + orthogonality analyses | ~200 MB (Git LFS) |
| `results/` | Full experimental results for Experiments 3 & 4 (steering, generalization, joint steering) | ~100 MB |

> **Note:** `.pt` files are tracked with [Git LFS](https://git-lfs.github.com/). After cloning, run `git lfs pull` if the large files were not downloaded automatically.

### Quick Start: Run Steering with Pre-trained Probes

If you only want to reproduce the steering experiments (Experiment 3) without re-generating prompts or re-training probes:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run activation steering on a model (probes are auto-discovered from data/probes/)
python -m src.experiments.03_steer_activations \
    --model_name Qwen/Qwen3-14B \
    --probes_dir data/probes \
    --output_dir results/exp3_steering \
    --alphas -1.0 0.0 1.0 \
    --max_samples 50 \
    --use_chat_template
```

Similarly, to run cross-task generalization and joint steering (Experiment 4):

```bash
python -m src.experiments.04_generalization \
    --model_name Qwen/Qwen3-14B \
    --probes_dir data/probes \
    --output_dir results/exp4_generalization \
    --run_joint_steering \
    --joint_dimensions all \
    --joint_global_alphas 0.0 1.0 \
    --max_samples 16 \
    --use_chat_template
```

See `examples/` for full command-line options for each experiment.

## Reproducing Experiments from Scratch

The sections below describe how to reproduce the entire pipeline **from scratch**, starting from prompt generation. If you want to use the pre-included data instead, see [Quick Start](#quick-start-run-steering-with-pre-trained-probes) above.

### Experiment 1: Generate Prompt Pairs

```bash
bash examples/run_exp1_generate_prompts.sh [NUM_SAMPLES] [SEED]
```

Generates binary-contrast prompt pairs for each metacognitive dimension using GSM8K and MMLU-Pro as base tasks. Output: `data/prompts/dim_*.json`.

### Experiment 2: Train Linear Probes

```bash
bash examples/run_exp2_train_probes.sh [MODEL_NAME]
# e.g., bash examples/run_exp2_train_probes.sh Qwen/Qwen3-14B
```

Extracts residual stream activations, trains per-layer logistic regression probes, and analyzes probe direction orthogonality. Output: `data/probes/probe_*.pt`.

### Experiment 3: Activation Steering

```bash
bash examples/run_exp3_steer.sh [MODEL_NAME]
```

Steers model behavior by injecting or suppressing probe directions in the residual stream. Evaluates on GSM8K with alpha = {-1, 0, +1}. Output: `results/exp3_steering/steer_*.json`.

### Experiment 4: Cross-Task Generalization & Joint Steering

```bash
bash examples/run_exp4_generalization.sh [MODEL_NAME]
```

Tests whether probes transfer to SimpleQA (unseen domain). Optionally runs joint six-dimensional steering. Output: `results/exp4_generalization/` and `results/exp4_joint/`.

## Pre-trained Probes

We provide pre-trained probe weights for all five models in `data/probes/`. These can be used directly for steering experiments (Experiments 3 and 4) without re-running Experiment 2.