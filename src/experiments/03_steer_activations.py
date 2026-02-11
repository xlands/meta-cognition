"""
实验三：激活值 Steering

用实验二训练好的探针方向进行 activation steering，验证探针捕获了因果有效的方向。

方法：
    h_l' = h_l + alpha * w
    其中 w 是探针的权重向量，alpha 控制干预强度。

实验设置：
    alpha = { -1, 0, 1 }  三组对照
    - alpha = 0   : baseline（无干预）
    - alpha = 1   : 正向 steer（激活对应元认知状态）
    - alpha = -1  : 反向 steer（抑制对应元认知状态）
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
    """获取模型的 transformer 层列表"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Unable to locate transformer layers for this model.")


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vec)
    return vec / norm if norm.item() > 0 else vec


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


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
) -> Tuple[int, np.ndarray, Dict]:
    """
    加载探针文件，返回 (best_layer, coef_vector, meta)。
    """
    payload = torch.load(probe_file, map_location="cpu", weights_only=False)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"Probe file has no results: {probe_file}")

    if layer_index is None:
        best = max(results, key=lambda r: r["accuracy"])
        layer_index = int(best["layer"])
        coef = best["coef"]
    else:
        match = [r for r in results if int(r["layer"]) == layer_index]
        if not match:
            raise ValueError(f"Layer {layer_index} not found in probe.")
        coef = match[0]["coef"]

    return layer_index, np.asarray(coef, dtype=np.float32), payload.get("meta", {})


def discover_probes(probes_dir: str, model_slug: str) -> Dict[str, str]:
    """
    发现所有探针文件，返回 {dimension_name: probe_file_path}。
    """
    probes = {}
    for fname in sorted(os.listdir(probes_dir)):
        if fname.startswith("probe_") and fname.endswith(".pt") and model_slug in fname:
            # probe_{dim}_{model}.pt
            dim_name = fname.replace("probe_", "").replace(f"_{model_slug}.pt", "")
            probes[dim_name] = os.path.join(probes_dir, fname)
    return probes


# ============================================================================
# Steering Hook
# ============================================================================

def make_steering_hook(
    vector: torch.Tensor,
    alpha: float,
    target_position: str = "last",
):
    """
    创建 forward hook，在指定层注入 steering 向量。

    h' = h + alpha * w
    """
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            return output

        delta = vector.to(hidden.device, dtype=hidden.dtype) * alpha

        if target_position == "all":
            hidden = hidden + delta
        else:
            # 只修改 last token
            attention_mask = None
            if len(inputs) > 1 and torch.is_tensor(inputs[1]):
                attention_mask = inputs[1]

            hidden = hidden.clone()
            if attention_mask is not None and attention_mask.dim() == 2:
                last_idx = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(hidden.size(0), device=hidden.device)
                hidden[batch_idx, last_idx] += delta
            else:
                hidden[:, -1, :] += delta

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook


# ============================================================================
# 生成
# ============================================================================

def format_prompts(
    tokenizer,
    prompts: List[str],
    use_chat_template: bool = True,
    system_prompt: str = "",
    enable_thinking: bool = False,
) -> List[str]:
    if not use_chat_template:
        return list(prompts)
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


def parse_thinking_output(tokenizer, output_ids: List[int]) -> Tuple[str, str]:
    """解析 Qwen3 thinking 输出"""
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return thinking, content


def run_generation(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_thinking: bool = False,
) -> Tuple[List[str], List[str], List[int]]:
    """批量生成，返回 (responses, thinking_contents, token_counts)"""
    responses = []
    thinking_contents = []
    token_counts = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[start : start + batch_size]
        tokenized = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)

        attention_mask = tokenized.get("attention_mask")
        input_lengths = (
            attention_mask.sum(dim=1) if attention_mask is not None
            else torch.full((tokenized["input_ids"].size(0),), tokenized["input_ids"].size(1), device=model.device)
        )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated = model.generate(**tokenized, **gen_kwargs)

        for i in range(len(generated)):
            out_ids = generated[i][int(input_lengths[i]):].tolist()
            token_counts.append(len(out_ids))

            if enable_thinking:
                thinking, content = parse_thinking_output(tokenizer, out_ids)
                thinking_contents.append(thinking)
                responses.append(content)
            else:
                responses.append(tokenizer.decode(out_ids, skip_special_tokens=True))
                thinking_contents.append("")

    return responses, thinking_contents, token_counts


# ============================================================================
# 行为指标
# ============================================================================

REFUSAL_PATTERNS = [
    r"\bi cannot\b", r"\bI can't\b", r"\bI won't\b", r"\bI will not\b",
    r"\bI am unable\b", r"\bas an AI\b", r"\bsorry\b", r"\bI cannot assist\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bI think\b", r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b",
    r"\bmight\b", r"\bI'm not sure\b", r"\bpossibly\b",
]

FORMAL_PATTERNS = [
    r"\btherefore\b", r"\bhence\b", r"\bthus\b", r"\bconsequently\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin conclusion\b",
]


def count_pattern_matches(text: str, patterns: List[str]) -> int:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    return sum(len(p.findall(text)) for p in compiled)


def compute_metrics(responses: List[str], thinking_contents: List[str] = None) -> Dict:
    """计算行为指标"""
    n = len(responses)
    if n == 0:
        return {}

    word_counts = [len(r.split()) for r in responses]
    combined = responses
    if thinking_contents:
        combined = [f"{t} {r}" for t, r in zip(thinking_contents, responses)]

    metrics = {
        "avg_word_count": float(np.mean(word_counts)),
        "median_word_count": float(np.median(word_counts)),
        "refusal_rate": sum(1 for r in responses if any(
            re.search(p, r, re.IGNORECASE) for p in REFUSAL_PATTERNS
        )) / n,
        "uncertainty_rate": sum(1 for r in combined if any(
            re.search(p, r, re.IGNORECASE) for p in UNCERTAINTY_PATTERNS
        )) / n,
        "formal_rate": sum(1 for r in combined if any(
            re.search(p, r, re.IGNORECASE) for p in FORMAL_PATTERNS
        )) / n,
    }

    return metrics


def extract_gsm8k_answer(text: str) -> Optional[str]:
    if "####" in text:
        return text.split("####")[-1].strip()
    return None


def extract_last_number(text: str) -> Optional[str]:
    matches = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return matches[-1] if matches else None


def compute_accuracy(responses: List[str], answers: List[str]) -> float:
    correct = 0
    total = 0
    for response, answer in zip(responses, answers):
        gold = extract_gsm8k_answer(answer) or answer.strip()
        gold_num = extract_last_number(gold)
        pred_num = extract_last_number(response)
        if gold_num and pred_num and gold_num.strip() == pred_num.strip():
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ============================================================================
# 数据加载
# ============================================================================

def load_steering_prompts(
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: str = "test",
    prompt_field: str = "question",
    answer_field: str = "answer",
    max_samples: int = 200,
    seed: int = 42,
    skip_samples: int = 0,
) -> Tuple[List[str], List[str]]:
    """加载 steering 用的测试数据

    Args:
        skip_samples: 跳过前 N 个样本（在 shuffle 之后），用于数据并行分片。
    """
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)
    # 先跳过，再截取
    total = len(dataset)
    start = min(skip_samples, total)
    end = min(start + max_samples, total) if max_samples > 0 else total
    dataset = dataset.select(range(start, end))

    prompts = [str(row[prompt_field]).strip() for row in dataset]
    answers = [str(row.get(answer_field, "")).strip() for row in dataset]
    return prompts, answers


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="实验三：激活值 Steering")

    # 模型
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="使用 4bit 量化加载模型（节省显存，适用于超大模型）")

    # 探针
    parser.add_argument("--probes_dir", type=str, default="data/probes")
    parser.add_argument("--probe_file", type=str, default="",
                       help="指定单个探针文件（覆盖 probes_dir 自动发现）")
    parser.add_argument("--dimension", type=str, default="all",
                       help="要 steer 的维度，逗号分隔，或 'all'")
    parser.add_argument("--layer_index", type=int, default=-1,
                       help="指定层索引，-1 表示使用最佳层")

    # Steering 参数
    parser.add_argument("--alphas", type=str, default="-1.0,0.0,1.0")
    parser.add_argument("--normalize_vector", action="store_true")
    parser.add_argument("--target_position", type=str, default="last",
                       choices=["last", "all"])

    # 数据
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--prompt_field", type=str, default="question")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--skip_samples", type=int, default=0,
                       help="跳过前 N 个样本（shuffle 后），用于数据并行分片")
    parser.add_argument("--seed", type=int, default=42)

    # 生成
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="")

    # 输出
    parser.add_argument("--output_dir", type=str, default="data/steering")
    parser.add_argument("--output_suffix", type=str, default="",
                       help="输出文件名后缀，用于数据并行时区分分片，如 '_shard0'")

    args = parser.parse_args()
    alphas = parse_float_list(args.alphas)
    model_slug = args.model_name.rstrip("/").split("/")[-1]
    trust_remote = args.trust_remote_code or should_trust_remote_code(args.model_name)
    torch_dtype = resolve_torch_dtype(args.dtype)

    # 发现探针
    if args.probe_file:
        dim_name = args.dimension if args.dimension != "all" else "custom"
        probe_files = {dim_name: args.probe_file}
    else:
        probe_files = discover_probes(args.probes_dir, model_slug)
        if args.dimension != "all":
            selected = [d.strip() for d in args.dimension.split(",")]
            probe_files = {k: v for k, v in probe_files.items() if k in selected}

    if not probe_files:
        raise FileNotFoundError(f"No probe files found in {args.probes_dir} for model {model_slug}")
    print(f"Found probes for dimensions: {list(probe_files.keys())}")

    # 加载模型
    total_start = time.time()
    print(f"\nLoading model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=trust_remote)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    print(f"Model loaded in {format_time(time.time() - total_start)}")

    # 加载测试数据
    prompts, answers = load_steering_prompts(
        args.dataset_name, args.dataset_config, args.split,
        args.prompt_field, args.answer_field, args.max_samples, args.seed,
        skip_samples=args.skip_samples,
    )
    formatted_prompts = format_prompts(
        tokenizer, prompts, args.use_chat_template, args.system_prompt, args.enable_thinking,
    )
    print(f"Loaded {len(prompts)} test prompts")

    layers = get_transformer_layers(model)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 逐维度 Steering ----
    for dim_name, probe_path in probe_files.items():
        print(f"\n{'=' * 60}")
        print(f"Steering dimension: {dim_name}")
        print(f"{'=' * 60}")

        layer_idx = None if args.layer_index < 0 else args.layer_index
        best_layer, probe_vec, probe_meta = load_probe_vector(probe_path, layer_idx)
        if layer_idx is None:
            layer_idx = best_layer

        vector = torch.tensor(probe_vec, dtype=torch.float32)
        if args.normalize_vector:
            vector = normalize_vector(vector)

        print(f"Using probe from layer {layer_idx} (accuracy: {probe_meta.get('best_accuracy', '?')})")
        print(f"Alphas: {alphas}")

        dim_results = []

        for alpha in alphas:
            alpha_start = time.time()
            print(f"\n--- Alpha = {alpha} ---")

            # 注册 hook
            hooks = []
            if alpha != 0.0:
                hook = layers[layer_idx].register_forward_hook(
                    make_steering_hook(vector, alpha, args.target_position)
                )
                hooks.append(hook)

            # 生成
            responses, thinking_contents, token_counts = run_generation(
                model, tokenizer, formatted_prompts,
                args.batch_size, args.max_new_tokens,
                args.do_sample, args.temperature, args.top_p,
                args.enable_thinking,
            )

            # 移除 hooks
            for h in hooks:
                h.remove()

            # 计算通用指标
            metrics = compute_metrics(responses, thinking_contents)
            metrics["avg_token_count"] = float(np.mean(token_counts))

            # 如果有 ground truth，计算准确率
            if answers and any(a for a in answers):
                metrics["accuracy"] = compute_accuracy(responses, answers)

            # === 维度专用评分 ===
            dim_scores_batch = compute_dimension_scores_batch(responses, dim_name)
            dim_score_summary = {}
            for metric_key, values in dim_scores_batch.items():
                dim_score_summary[f"dim_{metric_key}_mean"] = float(np.mean(values))
                dim_score_summary[f"dim_{metric_key}_std"] = float(np.std(values))
            metrics.update(dim_score_summary)

            composite_vals = dim_scores_batch.get("composite_score", [])
            if composite_vals:
                metrics["dim_composite_mean"] = float(np.mean(composite_vals))
                metrics["dim_composite_std"] = float(np.std(composite_vals))

            print(f"  Generic Metrics: avg_words={metrics['avg_word_count']:.0f}, "
                  f"refusal={metrics['refusal_rate']:.2f}, formal={metrics['formal_rate']:.2f}")
            print(f"  Dimension Score ({dim_name}): "
                  f"composite={metrics.get('dim_composite_mean', 0):.3f} "
                  f"± {metrics.get('dim_composite_std', 0):.3f}")
            if "accuracy" in metrics:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Time: {format_time(time.time() - alpha_start)}")

            dim_results.append({
                "alpha": alpha,
                "layer_index": layer_idx,
                "metrics": metrics,
                "responses_sample": [
                    {"prompt": p, "response": r, "thinking": t}
                    for p, r, t in zip(prompts[:10], responses[:10], thinking_contents[:10])
                ],
            })

        # 打印 steering 效果对比
        print(f"\n  >>> Steering Effect Summary for {dim_name}:")
        for result in dim_results:
            a = result["alpha"]
            cs = result["metrics"].get("dim_composite_mean", 0)
            wc = result["metrics"].get("avg_word_count", 0)
            acc = result["metrics"].get("accuracy", None)
            acc_str = f", acc={acc:.4f}" if acc is not None else ""
            print(f"      alpha={a:+5.1f}  composite={cs:+.3f}  avg_words={wc:.0f}{acc_str}")

        # 保存该维度的结果
        output_payload = {
            "meta": {
                "dimension": dim_name,
                "model_name": args.model_name,
                "probe_file": probe_path,
                "layer_index": layer_idx,
                "alphas": alphas,
                "target_position": args.target_position,
                "normalize_vector": args.normalize_vector,
                "dataset": args.dataset_name,
                "num_samples": len(prompts),
            },
            "results": dim_results,
        }

        suffix = args.output_suffix if args.output_suffix else ""
        out_path = os.path.join(args.output_dir, f"steer_{dim_name}_{model_slug}{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved → {out_path}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"实验三完成！总耗时: {format_time(total_time)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
