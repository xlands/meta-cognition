"""
实验二：提取 Prompt Last Residual，训练线性探针，分析正交性

流程：
1. 加载实验一生成的 Prompt Pairs
2. 加载语言模型，对每个 prompt 做 forward pass
3. 提取 prompt last token 的 hidden state（所有层）
4. 对每个维度，逐层训练 LogisticRegression 探针
5. PCA 分解 + 余弦相似度分析，证明6个维度的正交性

输出：
- data/probes/probe_{dim}_{model}.pt          — 每个维度的探针权重
- data/probes/orthogonality_analysis.json      — 正交性分析结果
- data/activations/activations_{dim}_{model}.pt — (可选) 激活值
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


# ============================================================================
# 数据加载
# ============================================================================

def load_dimension_prompts(prompts_dir: str, dim_name: str, max_pairs: int = 0) -> Tuple[List[str], np.ndarray]:
    """
    加载一个维度的 prompt pairs，展平为 prompts + labels。

    Args:
        prompts_dir: prompts 目录
        dim_name: 维度名
        max_pairs: 最多取多少对 prompt pair，0 表示全部

    Returns:
        prompts: [pos_0, neg_0, pos_1, neg_1, ...] 交替排列
        labels:  [1, 0, 1, 0, ...] 对应标签
    """
    path = os.path.join(prompts_dir, f"dim_{dim_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    pairs = payload["data"]
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    prompts = []
    labels = []
    for pair in pairs:
        prompts.append(pair["positive_prompt"])
        labels.append(1)
        prompts.append(pair["negative_prompt"])
        labels.append(0)

    return prompts, np.array(labels, dtype=np.int64)


def discover_dimensions(prompts_dir: str) -> List[str]:
    """自动发现 prompts 目录下所有维度文件"""
    dims = []
    for fname in sorted(os.listdir(prompts_dir)):
        if fname.startswith("dim_") and fname.endswith(".json"):
            dim_name = fname[4:-5]  # strip 'dim_' and '.json'
            dims.append(dim_name)
    return dims


# ============================================================================
# 激活值提取
# ============================================================================

def format_prompts_with_chat_template(
    tokenizer,
    prompts: List[str],
    system_prompt: str = "",
    enable_thinking: bool = False,
) -> List[str]:
    """用 chat template 格式化 prompts"""
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


def extract_last_token_activations(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """
    提取所有 transformer 层的 prompt last token hidden state。

    Returns:
        activations: List[Tensor], 每层一个 (num_samples, hidden_dim) 的 tensor
    """
    layer_buffers = None

    total = len(prompts)
    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc="Extracting activations"):
            batch = prompts[start : start + batch_size]

            tokenized = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

            outputs = model(
                **tokenized,
                output_hidden_states=True,
                use_cache=False,
            )

            # hidden_states[0] = embedding output, [1:] = transformer layers
            hidden_states = outputs.hidden_states[1:]

            if layer_buffers is None:
                layer_buffers = [[] for _ in range(len(hidden_states))]

            # 找到每个样本的 last token 位置
            attention_mask = tokenized["attention_mask"]
            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(attention_mask.size(0), device=model.device)

            for layer_idx, layer_state in enumerate(hidden_states):
                last_token_state = layer_state[batch_indices, last_indices]
                layer_buffers[layer_idx].append(
                    last_token_state.detach().cpu().to(torch.float16)
                )

    activations = [torch.cat(chunks, dim=0) for chunks in layer_buffers]
    return activations


# ============================================================================
# 探针训练
# ============================================================================

def train_probe_for_layer(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 42,
) -> Dict:
    """在单层上训练线性探针"""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 处理异常值
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # 标准化
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    clf = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000,
        class_weight="balanced", solver="liblinear",
        random_state=seed,
    )
    clf.fit(X_train_norm, y_train)

    preds = clf.predict(X_test_norm)
    acc = accuracy_score(y_test, preds)

    # 将系数转换回原始空间（撤销标准化）
    coef_norm = clf.coef_[0].astype(np.float32)
    intercept = float(clf.intercept_[0])
    coef = (coef_norm / std.squeeze(0)).astype(np.float32)
    intercept = float(intercept - np.dot(coef, mean.squeeze(0)))

    return {
        "coef": coef,
        "intercept": intercept,
        "accuracy": acc,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def train_probes_all_layers(
    activations: List[torch.Tensor],
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> List[Dict]:
    """对所有层训练探针"""
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels,
    )

    results = []
    for layer_idx, layer_acts in enumerate(activations):
        X = layer_acts.cpu().numpy().astype(np.float64)
        result = train_probe_for_layer(X, labels, train_idx, test_idx, seed)
        result["layer"] = layer_idx
        results.append(result)

        if (layer_idx + 1) % 5 == 0 or layer_idx == len(activations) - 1:
            print(f"  Layer {layer_idx:3d}: accuracy = {result['accuracy']:.4f}")

    return results


# ============================================================================
# 正交性分析
# ============================================================================

def orthogonality_analysis(
    probe_vectors: Dict[str, np.ndarray],
    all_activations: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict:
    """
    分析不同维度探针方向的正交性。

    1. 探针向量余弦相似度矩阵
    2. 激活差异向量的 PCA 分析（如果提供了激活值数据）

    Args:
        probe_vectors: {dim_name: coef_vector}
        all_activations: {dim_name: (pos_activations, neg_activations)} 可选
    """
    dim_names = sorted(probe_vectors.keys())
    n_dims = len(dim_names)

    # --- 1. 探针向量余弦相似度 ---
    vectors = np.stack([probe_vectors[name] for name in dim_names])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    normalized = vectors / norms
    cosine_matrix = (normalized @ normalized.T).tolist()

    result = {
        "dimensions": dim_names,
        "probe_cosine_similarity": cosine_matrix,
    }

    # --- 2. 激活差异的 PCA ---
    if all_activations and len(all_activations) >= 2:
        diff_vectors = []
        diff_labels = []
        for dim_name in dim_names:
            if dim_name not in all_activations:
                continue
            pos_acts, neg_acts = all_activations[dim_name]
            diff = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
            diff_vectors.append(diff)
            diff_labels.append(dim_name)

        if len(diff_vectors) >= 2:
            diff_matrix = np.stack(diff_vectors)
            n_components = min(len(diff_vectors), 6)
            pca = PCA(n_components=n_components)
            projected = pca.fit_transform(diff_matrix)

            result["activation_diff_pca"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "projected_2d": {
                    name: projected[i, :2].tolist()
                    for i, name in enumerate(diff_labels)
                },
            }

            # 激活差异向量的余弦相似度
            diff_norms = np.linalg.norm(diff_matrix, axis=1, keepdims=True) + 1e-8
            diff_normalized = diff_matrix / diff_norms
            diff_cosine = (diff_normalized @ diff_normalized.T).tolist()
            result["activation_diff_cosine_similarity"] = diff_cosine

    return result


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="实验二：提取激活值，训练探针，分析正交性"
    )

    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--system_prompt", type=str, default="")

    # 输入参数
    parser.add_argument("--prompts_dir", type=str, default="data/prompts")
    parser.add_argument(
        "--dimensions", type=str, default="all",
        help="要处理的维度，逗号分隔，或 'all'",
    )

    # 提取参数
    parser.add_argument("--max_pairs", type=int, default=0,
                       help="每个维度最多用多少对 prompt pair，0 表示全部")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="使用 4bit 量化加载模型（节省显存，适用于超大模型）")

    # 探针参数
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/probes")
    parser.add_argument("--save_activations", action="store_true")
    parser.add_argument("--activations_dir", type=str, default="data/activations")

    args = parser.parse_args()

    # 确定维度
    if args.dimensions == "all":
        dims_to_process = discover_dimensions(args.prompts_dir)
    else:
        dims_to_process = [d.strip() for d in args.dimensions.split(",")]

    if not dims_to_process:
        raise FileNotFoundError(f"No dimension files found in {args.prompts_dir}")
    print(f"Dimensions to process: {dims_to_process}")

    # 加载模型
    trust_remote = args.trust_remote_code or should_trust_remote_code(args.model_name)
    torch_dtype = resolve_torch_dtype(args.dtype)

    print(f"\nLoading model: {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=trust_remote,
    )
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
    print(f"Model loaded. Device: {model.device}")

    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_activations:
        os.makedirs(args.activations_dir, exist_ok=True)

    model_slug = args.model_name.rstrip("/").split("/")[-1]

    # 用于正交性分析的收集器
    all_probe_vectors = {}       # dim -> best_layer_coef
    all_activation_diffs = {}    # dim -> (pos_mean, neg_mean) at best layer

    # ---- 逐维度处理 ----
    for dim_name in dims_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing dimension: {dim_name}")
        print(f"{'=' * 60}")

        # 加载数据
        prompts, labels = load_dimension_prompts(args.prompts_dir, dim_name, args.max_pairs)
        print(f"Loaded {len(prompts)} prompts (pos={labels.sum()}, neg={len(labels)-labels.sum()})")

        # 格式化
        if args.use_chat_template:
            prompts = format_prompts_with_chat_template(
                tokenizer, prompts, args.system_prompt, args.enable_thinking,
            )

        # 提取激活值
        print("\nExtracting prompt last-token activations...")
        activations = extract_last_token_activations(
            model, tokenizer, prompts, args.batch_size, args.max_length,
        )
        print(f"Extracted {len(activations)} layers, shape per layer: {activations[0].shape}")

        # 保存激活值（可选）
        if args.save_activations:
            act_path = os.path.join(
                args.activations_dir, f"activations_{dim_name}_{model_slug}.pt",
            )
            torch.save({
                "activations": activations,
                "labels": torch.tensor(labels, dtype=torch.long),
            }, act_path)
            print(f"Saved activations → {act_path}")

        # 训练探针
        print("\nTraining linear probes (all layers)...")
        probe_results = train_probes_all_layers(
            activations, labels, args.test_size, args.seed,
        )

        # 找到最佳层
        best = max(probe_results, key=lambda r: r["accuracy"])
        print(f"\n*** Best layer: {best['layer']}  accuracy: {best['accuracy']:.4f} ***")

        # 收集最佳层探针向量
        all_probe_vectors[dim_name] = best["coef"]

        # 收集最佳层的激活值用于正交性分析
        best_layer_acts = activations[best["layer"]].cpu().numpy().astype(np.float32)
        pos_mask = labels == 1
        neg_mask = labels == 0
        all_activation_diffs[dim_name] = (
            best_layer_acts[pos_mask],
            best_layer_acts[neg_mask],
        )

        # 保存探针
        probe_path = os.path.join(args.output_dir, f"probe_{dim_name}_{model_slug}.pt")
        torch.save({
            "results": probe_results,
            "meta": {
                "dimension": dim_name,
                "model_name": args.model_name,
                "num_samples": len(labels),
                "best_layer": best["layer"],
                "best_accuracy": best["accuracy"],
            },
            "config": {
                "test_size": args.test_size,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
            },
        }, probe_path)
        print(f"Saved probes → {probe_path}")

        # 保存摘要
        summary = {
            "dimension": dim_name,
            "model_name": args.model_name,
            "best_layer": best["layer"],
            "best_accuracy": best["accuracy"],
            "num_layers": len(probe_results),
            "num_samples": len(labels),
            "layer_accuracies": {r["layer"]: r["accuracy"] for r in probe_results},
        }
        summary_path = probe_path + ".summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary → {summary_path}")

    # ---- 正交性分析 ----
    if len(all_probe_vectors) >= 2:
        print(f"\n{'=' * 60}")
        print("Orthogonality Analysis")
        print(f"{'=' * 60}")

        analysis = orthogonality_analysis(all_probe_vectors, all_activation_diffs)

        # 打印余弦相似度矩阵
        dim_names = analysis["dimensions"]
        cosine_mat = analysis["probe_cosine_similarity"]
        print("\nProbe vector cosine similarity matrix:")
        header = "            " + "  ".join(f"{d[:8]:>8s}" for d in dim_names)
        print(header)
        for i, name in enumerate(dim_names):
            row = "  ".join(f"{cosine_mat[i][j]:8.4f}" for j in range(len(dim_names)))
            print(f"{name[:10]:>10s}  {row}")

        if "activation_diff_pca" in analysis:
            pca_info = analysis["activation_diff_pca"]
            print(f"\nActivation diff PCA explained variance: {pca_info['explained_variance_ratio']}")

        analysis_path = os.path.join(args.output_dir, f"orthogonality_analysis_{model_slug}.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved orthogonality analysis → {analysis_path}")

    print(f"\n{'=' * 60}")
    print("实验二完成！探针训练 + 正交性分析已保存。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
