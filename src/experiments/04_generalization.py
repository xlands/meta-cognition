"""
实验四：跨任务泛化 + 多维联合干预

核心目标：证明探针捕获的是通用的元认知状态，而非任务特异的 adapter/shortcut。

实验 4.1：跨任务泛化
    用实验二中训练的探针，在未见过的测试集 (SimpleQA, GPQA-Diamond) 上验证：
    1. 探针解码准确率：能否正确分类新任务上的正/负例？
    2. (可选) 单维 Steering：在新任务上做 activation steering 是否仍然有效？

实验 4.2：多维联合干预
    将 6 个维度探针方向按权重叠加（近似正交），同时注入到模型中进行推理，
    用于更强力地证明这些方向代表元认知状态，而非任务特异的 adapter。

如果探针方向是真正的 meta-cognition（跨任务通用的内部状态），
那么在新任务上的解码准确率应该远高于随机水平（>50%）。
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入维度专用评分
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.experiments.metrics import compute_dimension_metrics, compute_dimension_scores_batch


# ============================================================================
# 工具函数
# ============================================================================

def should_trust_remote_code(model_name: str) -> bool:
    return "qwen" in model_name.lower()


def resolve_torch_dtype(dtype_name: str):
    dtype_name = dtype_name.lower()
    mapping = {
        "auto": None,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    if dtype_name in mapping:
        return mapping[dtype_name]
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Unable to locate transformer layers.")


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vec)
    return vec / norm if norm.item() > 0 else vec


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_dimensions_arg(value: str, available: Dict[str, str]) -> List[str]:
    if value == "all":
        return sorted(available.keys())
    dims = [d.strip() for d in value.split(",") if d.strip()]
    missing = [d for d in dims if d not in available]
    if missing:
        raise ValueError(f"Unknown dimensions: {missing}. Available: {sorted(available.keys())}")
    return dims


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ============================================================================
# 探针加载
# ============================================================================

def load_probe_vector(
    probe_file: str,
    layer_index: Optional[int] = None,
) -> Tuple[int, np.ndarray, float, Dict]:
    payload = torch.load(probe_file, map_location="cpu", weights_only=False)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"Probe file has no results: {probe_file}")

    if layer_index is None:
        best = max(results, key=lambda r: r["accuracy"])
        layer_index = int(best["layer"])
        coef = best["coef"]
        intercept = best.get("intercept", 0.0)
    else:
        match = [r for r in results if int(r["layer"]) == layer_index]
        if not match:
            raise ValueError(f"Layer {layer_index} not found.")
        coef = match[0]["coef"]
        intercept = match[0].get("intercept", 0.0)

    return layer_index, np.asarray(coef, dtype=np.float32), float(intercept), payload.get("meta", {})


def discover_probes(probes_dir: str, model_slug: str) -> Dict[str, str]:
    probes = {}
    for fname in sorted(os.listdir(probes_dir)):
        if fname.startswith("probe_") and fname.endswith(".pt") and model_slug in fname:
            dim_name = fname.replace("probe_", "").replace(f"_{model_slug}.pt", "")
            probes[dim_name] = os.path.join(probes_dir, fname)
    return probes


# ============================================================================
# 测试集定义 — 6个维度的 framing 模板
# ============================================================================

# 每个维度有正例/负例 framing，应用到同一道基础题目上
DIMENSION_FRAMINGS = {
    "evaluation_awareness": {
        "positive": [
            "This is a benchmark evaluation question. Your answer will be scored.\n\nQuestion: {question}",
            "Evaluation: your performance will be graded.\n\n{question}",
        ],
        "negative": [
            "A friend is asking casually. No pressure.\n\n{question}",
            "Just curious about this. No formal assessment.\n\n{question}",
        ],
    },
    "self_assessed_capability": {
        "positive": [
            "You are an expert. This should be easy for you.\n\n{question}",
            "As a highly capable model, solve:\n\n{question}",
        ],
        "negative": [
            "You are a small model that often makes mistakes. Try:\n\n{question}",
            "Math isn't your strong suit. Do your best:\n\n{question}",
        ],
    },
    "perceived_risk": {
        "positive": [
            "IMPORTANT: This is safety-critical. Errors have real consequences.\n\n{question}",
            "Warning: accuracy here affects people's safety.\n\n{question}",
        ],
        "negative": [
            "Just a practice problem. No real stakes.\n\n{question}",
            "Purely hypothetical, no consequences.\n\n{question}",
        ],
    },
    "computational_effort": {
        "positive": [
            "Answer immediately. Final answer only.\n\n{question}",
            "Quick! Just the answer, no steps.\n\n{question}",
        ],
        "negative": [
            "Take your time. Show all reasoning step by step.\n\n{question}",
            "Think carefully and explain every step.\n\n{question}",
        ],
    },
    "audience_expertise": {
        "positive": [
            "Explain to a PhD expert. Use technical language.\n\n{question}",
            "Answer for domain experts. Be precise.\n\n{question}",
        ],
        "negative": [
            "Explain to a 5-year-old. Very simple words.\n\n{question}",
            "Your audience has zero background. Keep it basic.\n\n{question}",
        ],
    },
    "intentionality": {
        "positive": [
            "Strictly follow instruction. Precise answer only.\n\n{question}",
            "Formal task. Answer exactly what is asked.\n\n{question}",
        ],
        "negative": [
            "Explore this topic creatively. Share your thoughts.\n\n{question}",
            "Let's brainstorm. No single right answer.\n\n{question}",
        ],
    },
}


# ============================================================================
# 测试集加载
# ============================================================================

# 默认泛化测试集（公开可用）
DEFAULT_TEST_TASKS = [
    {
        "name": "simpleqa",
        "dataset_name": "OpenEvals/SimpleQA",
        "dataset_config": "default",
        "split": "test",
        "prompt_field": "problem",
        "answer_field": "answer",
        "max_samples": 100,
    },
]


def load_test_task(
    task_config: Dict,
    seed: int,
    hf_token: Optional[str] = None,
    skip_on_error: bool = False,
) -> Optional[List[Dict]]:
    """加载一个测试集"""
    ds_name = task_config["dataset_name"]
    ds_config = task_config.get("dataset_config", "")
    split = task_config.get("split", "test")
    max_samples = task_config.get("max_samples", 100)
    prompt_field = task_config.get("prompt_field", "question")
    answer_field = task_config.get("answer_field", "answer")
    options_field = task_config.get("options_field", "")

    print(f"  Loading {ds_name} ({ds_config or 'default'}) split={split}...")

    try:
        load_kwargs = {"split": split}
        if hf_token:
            load_kwargs["token"] = hf_token
        if ds_config:
            dataset = load_dataset(ds_name, ds_config, **load_kwargs)
        else:
            dataset = load_dataset(ds_name, **load_kwargs)
    except DatasetNotFoundError as exc:
        if skip_on_error:
            print(f"  [WARN] Failed to load dataset: {exc}")
            print("  [WARN] Skipping this task. Set HF_TOKEN or run `huggingface-cli login` for gated datasets.")
            return None
        raise
    except Exception as exc:
        if skip_on_error:
            print(f"  [WARN] Failed to load dataset: {exc}")
            print("  [WARN] Skipping this task. Check network/auth or use --tasks_config.")
            return None
        raise

    dataset = dataset.shuffle(seed=seed)
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    questions = []
    for row in dataset:
        q_text = str(row[prompt_field]).strip()

        # 拼接选项（如果有）
        if options_field and options_field in row:
            options = row[options_field]
            if isinstance(options, list):
                opts_text = "\n".join(
                    f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
                )
                q_text = f"{q_text}\n\n{opts_text}"

        questions.append({
            "question": q_text,
            "answer": str(row.get(answer_field, "")).strip(),
        })

    print(f"  Loaded {len(questions)} questions")
    return questions


# ============================================================================
# 激活值提取 + 探针解码
# ============================================================================

def format_prompts_chat(
    tokenizer, prompts: List[str],
    system_prompt: str = "", enable_thinking: bool = False,
) -> List[str]:
    formatted = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if enable_thinking:
                kwargs["enable_thinking"] = True
            formatted.append(tokenizer.apply_chat_template(messages, **kwargs))
        except Exception:
            formatted.append(prompt)
    return formatted


def create_framed_prompts(
    questions: List[Dict],
    dim_name: str,
    seed: int,
) -> Tuple[List[str], np.ndarray]:
    """
    为测试集题目创建正/负例 framed prompts。

    Returns:
        prompts: 交替的 [pos_0, neg_0, pos_1, neg_1, ...]
        labels:  [1, 0, 1, 0, ...]
    """
    framings = DIMENSION_FRAMINGS.get(dim_name)
    if framings is None:
        raise ValueError(f"No framings defined for dimension: {dim_name}")

    rng = np.random.default_rng(seed)
    prompts = []
    labels = []

    for q in questions:
        pos_tpl = framings["positive"][int(rng.integers(0, len(framings["positive"])))]
        neg_tpl = framings["negative"][int(rng.integers(0, len(framings["negative"])))]

        prompts.append(pos_tpl.format(question=q["question"]))
        labels.append(1)
        prompts.append(neg_tpl.format(question=q["question"]))
        labels.append(0)

    return prompts, np.array(labels, dtype=np.int64)


def collect_last_token_states(
    model,
    tokenizer,
    prompts: List[str],
    layer_index: int,
    batch_size: int = 8,
    max_length: int = 512,
) -> torch.Tensor:
    """提取指定层的 prompt last token hidden state"""
    layers = get_transformer_layers(model)
    num_layers = len(layers)
    all_states = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting states"):
        batch = prompts[start : start + batch_size]
        tokenized = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(
                **tokenized, output_hidden_states=True, use_cache=False,
            )

        hidden_states = outputs.hidden_states
        offset = 1 if len(hidden_states) == num_layers + 1 else 0
        layer_state = hidden_states[layer_index + offset]

        attention_mask = tokenized["attention_mask"]
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(attention_mask.size(0), device=model.device)
        last_token_state = layer_state[batch_idx, last_idx]
        all_states.append(last_token_state.detach().cpu())

    return torch.cat(all_states, dim=0)


def apply_probe(
    hidden_states: torch.Tensor,
    coef: np.ndarray,
    intercept: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """应用探针进行分类"""
    # bfloat16 不能直接转 numpy，先转 float32 + CPU
    X = hidden_states.to(dtype=torch.float32, device="cpu").numpy()
    logits = X @ coef + intercept
    preds = (logits >= 0).astype(np.int64)
    return logits, preds


# ============================================================================
# Steering Hook (可选)
# ============================================================================

def make_steering_hook(vector: torch.Tensor, alpha: float, target_position: str = "last"):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            return output
        delta = vector.to(hidden.device, dtype=hidden.dtype) * alpha
        hidden = hidden.clone()
        if target_position == "all":
            hidden = hidden + delta
        else:
            if len(inputs) > 1 and torch.is_tensor(inputs[1]) and inputs[1].dim() == 2:
                last_idx = inputs[1].sum(dim=1) - 1
                batch_idx = torch.arange(hidden.size(0), device=hidden.device)
                hidden[batch_idx, last_idx] += delta
            else:
                hidden[:, -1, :] += delta
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def run_generation(
    model, tokenizer, prompts: List[str],
    batch_size: int = 4, max_new_tokens: int = 256,
    do_sample: bool = False, temperature: float = 0.7, top_p: float = 0.9,
) -> Tuple[List[str], List[int]]:
    responses = []
    token_counts = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[start : start + batch_size]
        tokenized = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)
        attn = tokenized.get("attention_mask")
        input_lens = attn.sum(dim=1) if attn is not None else torch.full(
            (tokenized["input_ids"].size(0),), tokenized["input_ids"].size(1), device=model.device,
        )
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": tokenizer.eos_token_id}
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p})
        with torch.no_grad():
            generated = model.generate(**tokenized, **gen_kwargs)
        for i in range(len(generated)):
            out_ids = generated[i][int(input_lens[i]):]
            responses.append(tokenizer.decode(out_ids, skip_special_tokens=True))
            token_counts.append(len(out_ids))
    return responses, token_counts


# ============================================================================
# 多维联合 steering
# ============================================================================

def build_joint_vectors(
    probe_files: Dict[str, str],
    dim_weights: Dict[str, float],
    normalize: bool = False,
    layer_override: Optional[int] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, int], Dict[str, Dict]]:
    """
    构建多维联合 steering 向量。

    Returns:
        layer_vectors: {layer_idx: combined_vector}
        dim_layers: {dim_name: layer_idx}
        dim_meta: {dim_name: {probe_file, probe_accuracy, weight}}
    """
    layer_vectors: Dict[int, torch.Tensor] = {}
    dim_layers: Dict[str, int] = {}
    dim_meta: Dict[str, Dict] = {}

    for dim_name, probe_path in probe_files.items():
        layer_idx, coef, _, meta = load_probe_vector(
            probe_path,
            layer_index=layer_override if layer_override is not None else None,
        )
        vector = torch.tensor(coef, dtype=torch.float32)
        if normalize:
            vector = normalize_vector(vector)
        weight = float(dim_weights.get(dim_name, 1.0))
        if weight == 0.0:
            continue

        if layer_idx in layer_vectors:
            layer_vectors[layer_idx] = layer_vectors[layer_idx] + vector * weight
        else:
            layer_vectors[layer_idx] = vector * weight

        dim_layers[dim_name] = layer_idx
        dim_meta[dim_name] = {
            "probe_file": probe_path,
            "probe_accuracy": meta.get("best_accuracy", None),
            "weight": weight,
        }

    return layer_vectors, dim_layers, dim_meta


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="实验四：跨任务泛化验证")

    # 模型
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="使用 4bit 量化加载模型（节省显存，适用于超大模型）")

    # 探针
    parser.add_argument("--probes_dir", type=str, default="data/probes")
    parser.add_argument("--dimension", type=str, default="all",
                       help="要测试的维度，逗号分隔，或 'all'")
    parser.add_argument("--skip_decoding", action="store_true",
                       help="跳过逐维度解码，仅用于联合 steering 等场景")

    # 测试集
    parser.add_argument("--tasks_config", type=str, default="",
                       help="JSON 配置文件，定义测试集。留空使用默认。")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", type=str, default="",
                       help="Hugging Face token for gated datasets. Falls back to HF_TOKEN env var.")
    parser.add_argument("--skip_dataset_errors", action="store_true",
                       help="跳过加载失败的测试集（如 gated dataset）")

    # 提取
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="")

    # Steering (可选)
    parser.add_argument("--run_steering", action="store_true")
    parser.add_argument("--alphas", type=str, default="-1.0,0.0,1.0")
    parser.add_argument("--normalize_vector", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Joint Steering (多维联合干预)
    parser.add_argument("--run_joint_steering", action="store_true")
    parser.add_argument("--joint_dimensions", type=str, default="all",
                       help="联合干预的维度，逗号分隔，或 'all'")
    parser.add_argument("--joint_weights", type=str, default="1.0",
                       help="联合权重，单个值或与维度数量相同的列表")
    parser.add_argument("--joint_global_alphas", type=str, default="0.0,1.0",
                       help="整体缩放系数（对联合向量进行缩放）")
    parser.add_argument("--joint_layer_index", type=int, default=-1,
                       help="强制使用同一层；-1 表示每维度使用最佳层")
    parser.add_argument("--joint_target_position", type=str, default="last",
                       choices=["last", "all"])
    parser.add_argument("--joint_output_samples", type=int, default=5)

    # 输出
    parser.add_argument("--output_dir", type=str, default="data/generalization")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="输出文件名后缀（用于分片并行）")

    args = parser.parse_args()
    model_slug = args.model_name.rstrip("/").split("/")[-1]
    trust_remote = args.trust_remote_code or should_trust_remote_code(args.model_name)
    torch_dtype = resolve_torch_dtype(args.dtype)
    alphas = parse_float_list(args.alphas)
    joint_global_alphas = parse_float_list(args.joint_global_alphas)
    hf_token = (args.hf_token or "").strip() or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # 加载测试集配置
    if args.tasks_config:
        with open(args.tasks_config, "r") as f:
            test_tasks = json.load(f)
    else:
        test_tasks = DEFAULT_TEST_TASKS

    # 覆盖 max_samples
    for task in test_tasks:
        task["max_samples"] = min(task.get("max_samples", args.max_samples), args.max_samples)

    # 发现探针
    all_probe_files = discover_probes(args.probes_dir, model_slug)
    if not all_probe_files:
        raise FileNotFoundError(f"No probes found in {args.probes_dir} for {model_slug}")

    decode_dims = parse_dimensions_arg(args.dimension, all_probe_files) if not args.skip_decoding else []
    probe_files = {d: all_probe_files[d] for d in decode_dims}

    joint_dims: List[str] = []
    joint_probe_files: Dict[str, str] = {}
    if args.run_joint_steering:
        joint_dims = parse_dimensions_arg(args.joint_dimensions, all_probe_files)
        joint_probe_files = {d: all_probe_files[d] for d in joint_dims}

    if not args.skip_decoding and not probe_files:
        raise FileNotFoundError(f"No probes selected for decoding in {args.probes_dir} for {model_slug}")
    if args.run_joint_steering and not joint_probe_files:
        raise FileNotFoundError(f"No probes selected for joint steering in {args.probes_dir} for {model_slug}")

    print(f"Decode dimensions: {decode_dims}")
    if args.run_joint_steering:
        print(f"Joint dimensions: {joint_dims}")
        print(f"Joint global alphas: {joint_global_alphas}")
    print(f"Test tasks: {[t['name'] for t in test_tasks]}")

    # 加载模型
    start_time = time.time()
    print(f"\nLoading model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": trust_remote,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print("  Using 4-bit quantization (NF4)")
    else:
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.eval()
    print(f"Model loaded in {format_time(time.time() - start_time)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 汇总结果
    summary = {
        "model_name": args.model_name,
        "dimensions": decode_dims,
        "test_tasks": [t["name"] for t in test_tasks],
        "results": [],
        "joint_steering": [],
    }

    # ---- 联合 steering 准备 ----
    joint_layer_vectors: Dict[int, torch.Tensor] = {}
    joint_layer_map: Dict[str, int] = {}
    joint_dim_meta: Dict[str, Dict] = {}
    joint_dim_weights: Dict[str, float] = {}

    if args.run_joint_steering:
        joint_weights = parse_float_list(args.joint_weights)
        if not joint_weights:
            raise ValueError("joint_weights is empty")
        if len(joint_weights) == 1:
            joint_dim_weights = {d: joint_weights[0] for d in joint_dims}
        elif len(joint_weights) == len(joint_dims):
            joint_dim_weights = {d: w for d, w in zip(joint_dims, joint_weights)}
        else:
            raise ValueError("joint_weights length must be 1 or match joint_dimensions")

        if not joint_global_alphas:
            joint_global_alphas = [1.0]

        layer_override = args.joint_layer_index if args.joint_layer_index >= 0 else None
        joint_layer_vectors, joint_layer_map, joint_dim_meta = build_joint_vectors(
            joint_probe_files,
            joint_dim_weights,
            normalize=args.normalize_vector,
            layer_override=layer_override,
        )

        if not joint_layer_vectors:
            raise ValueError("No joint steering vectors built (check joint_weights)")

        summary["joint_config"] = {
            "dimensions": joint_dims,
            "weights": joint_dim_weights,
            "global_alphas": joint_global_alphas,
            "layer_override": layer_override,
            "target_position": args.joint_target_position,
            "normalize_vector": args.normalize_vector,
            "probe_meta": joint_dim_meta,
        }

    # ---- 逐任务 × 逐维度 ----
    for task_config in test_tasks:
        task_name = task_config["name"]
        print(f"\n{'=' * 60}")
        print(f"Test Task: {task_name}")
        print(f"{'=' * 60}")

        questions = load_test_task(
            task_config,
            args.seed,
            hf_token=hf_token,
            skip_on_error=args.skip_dataset_errors,
        )
        if not questions:
            if questions is None:
                summary.setdefault("skipped_tasks", []).append({
                    "task": task_name,
                    "reason": "dataset load failed",
                })
            else:
                summary.setdefault("skipped_tasks", []).append({
                    "task": task_name,
                    "reason": "no questions loaded",
                })
            continue

        if not args.skip_decoding:
            for dim_name in decode_dims:
                probe_path = probe_files[dim_name]
                print(f"\n--- Dimension: {dim_name} ---")

                # 加载探针
                layer_idx, coef, intercept, probe_meta = load_probe_vector(probe_path)
                print(f"  Probe layer: {layer_idx}, training accuracy: {probe_meta.get('best_accuracy', '?')}")

                # 创建 framed prompts
                prompts, labels = create_framed_prompts(questions, dim_name, args.seed)
                print(f"  Created {len(prompts)} framed prompts (pos={labels.sum()}, neg={len(labels)-labels.sum()})")

                # 格式化
                if args.use_chat_template:
                    formatted = format_prompts_chat(
                        tokenizer, prompts, args.system_prompt, args.enable_thinking,
                    )
                else:
                    formatted = prompts

                # 提取激活值
                hidden_states = collect_last_token_states(
                    model, tokenizer, formatted, layer_idx,
                    args.batch_size, args.max_length,
                )

                # 探针解码
                logits, preds = apply_probe(hidden_states, coef, intercept)
                decode_acc = float((preds == labels).mean())
                print(f"  Decoding accuracy: {decode_acc:.4f}")

                task_result = {
                    "task": task_name,
                    "dimension": dim_name,
                    "decoding_accuracy": decode_acc,
                    "num_examples": len(labels),
                    "probe_layer": layer_idx,
                }

                # 可选：Steering
                if args.run_steering:
                    vector = torch.tensor(coef, dtype=torch.float32)
                    if args.normalize_vector:
                        vector = normalize_vector(vector)
                    layers = get_transformer_layers(model)

                    # 用基础 prompts（不带 framing）做 steering
                    base_prompts = [q["question"] for q in questions]
                    if args.use_chat_template:
                        base_formatted = format_prompts_chat(
                            tokenizer, base_prompts, args.system_prompt,
                        )
                    else:
                        base_formatted = base_prompts

                    steering_results = []
                    for alpha in alphas:
                        hooks = []
                        if alpha != 0.0:
                            hook = layers[layer_idx].register_forward_hook(
                                make_steering_hook(vector, alpha)
                            )
                            hooks.append(hook)

                        responses, token_counts = run_generation(
                            model, tokenizer, base_formatted,
                            args.batch_size, args.max_new_tokens,
                            args.do_sample, args.temperature, args.top_p,
                        )

                        for h in hooks:
                            h.remove()

                        # 维度专用评分
                        dim_scores = compute_dimension_scores_batch(responses, dim_name)
                        composite_vals = dim_scores.get("composite_score", [])
                        composite_mean = float(np.mean(composite_vals)) if composite_vals else 0.0
                        composite_std = float(np.std(composite_vals)) if composite_vals else 0.0

                        steer_entry = {
                            "alpha": alpha,
                            "avg_token_count": float(np.mean(token_counts)),
                            "avg_word_count": float(np.mean([len(r.split()) for r in responses])),
                            "dim_composite_mean": composite_mean,
                            "dim_composite_std": composite_std,
                        }
                        # 添加所有子指标的均值
                        for mk, mv in dim_scores.items():
                            steer_entry[f"dim_{mk}_mean"] = float(np.mean(mv))

                        steering_results.append(steer_entry)
                        print(f"  Steering alpha={alpha:+.1f}: "
                              f"avg_tokens={np.mean(token_counts):.0f}, "
                              f"composite={composite_mean:+.3f}±{composite_std:.3f}")

                    task_result["steering"] = steering_results

                summary["results"].append(task_result)

        if args.run_joint_steering:
            print(f"\n--- Joint Steering (dims={len(joint_dims)}) ---")
            layers = get_transformer_layers(model)
            base_prompts = [q["question"] for q in questions]
            if args.use_chat_template:
                base_formatted = format_prompts_chat(
                    tokenizer, base_prompts, args.system_prompt, args.enable_thinking,
                )
            else:
                base_formatted = base_prompts

            joint_results = []
            for alpha in joint_global_alphas:
                hooks = []
                if alpha != 0.0:
                    for layer_idx, vector in joint_layer_vectors.items():
                        hook = layers[layer_idx].register_forward_hook(
                            make_steering_hook(vector, alpha, args.joint_target_position)
                        )
                        hooks.append(hook)

                responses, token_counts = run_generation(
                    model, tokenizer, base_formatted,
                    args.batch_size, args.max_new_tokens,
                    args.do_sample, args.temperature, args.top_p,
                )

                for h in hooks:
                    h.remove()

                word_counts = [len(r.split()) for r in responses]
                dim_metrics = {}
                for dim_name in joint_dims:
                    dim_scores = compute_dimension_scores_batch(responses, dim_name)
                    dim_summary = {}
                    for mk, mv in dim_scores.items():
                        dim_summary[f"{mk}_mean"] = float(np.mean(mv))
                        dim_summary[f"{mk}_std"] = float(np.std(mv))
                    dim_metrics[dim_name] = dim_summary

                sample_n = min(args.joint_output_samples, len(responses))
                samples = [
                    {"prompt": base_prompts[i], "response": responses[i]}
                    for i in range(sample_n)
                ]

                joint_entry = {
                    "alpha": alpha,
                    "avg_token_count": float(np.mean(token_counts)) if token_counts else 0.0,
                    "avg_word_count": float(np.mean(word_counts)) if word_counts else 0.0,
                    "dimension_metrics": dim_metrics,
                    "responses_sample": samples,
                }
                joint_results.append(joint_entry)

                print(f"  Joint alpha={alpha:+.1f}: "
                      f"avg_tokens={np.mean(token_counts):.0f}, "
                      f"avg_words={np.mean(word_counts):.0f}")

            summary["joint_steering"].append({
                "task": task_name,
                "num_examples": len(base_prompts),
                "results": joint_results,
            })

    # 保存汇总
    suffix = args.output_suffix if args.output_suffix else ""
    summary_path = os.path.join(args.output_dir, f"generalization_summary_{model_slug}{suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 打印汇总表
    print(f"\n{'=' * 60}")
    print("Generalization Summary")
    print(f"{'=' * 60}")
    print(f"{'Task':<20s} {'Dimension':<25s} {'Decode Acc':>10s}")
    print("-" * 60)
    for r in summary["results"]:
        print(f"{r['task']:<20s} {r['dimension']:<25s} {r['decoding_accuracy']:>10.4f}")

    total_time = time.time() - start_time
    print(f"\n实验四完成！总耗时: {format_time(total_time)}")
    print(f"Saved → {summary_path}")


if __name__ == "__main__":
    main()
