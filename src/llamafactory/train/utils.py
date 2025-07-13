import json
import os
from pathlib import Path
from typing import Iterable, List, Union

import matplotlib.pyplot as plt
import torch


# ----------------------------- 1. plot_loss -----------------------------
def _load_log_history(output_dir: Union[str, Path]) -> List[dict]:
    """
    读取 trainer_state.json（或存在的 log_history*.json），返回 log_history 列表。
    """
    output_dir = Path(output_dir)
    state_file = output_dir / "trainer_state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"未找到 {state_file}，请确认训练目录是否正确。")

    with open(state_file, "r", encoding="utf-8") as f:
        trainer_state = json.load(f)

    return trainer_state.get("log_history", [])


def plot_loss(output_dir: Union[str, Path], keys: Iterable[str] = ("loss",)) -> None:
    """
    根据 trainer_state.json 中的 log_history 绘制 loss / accuracy 曲线。

    Args:
        output_dir (str|Path): 训练输出目录。
        keys (Iterable[str]): 需要绘制的字段名列表，例如
            ["loss", "eval_loss", "eval_accuracy"] ...
    """
    log_history = _load_log_history(output_dir)
    if not log_history:
        print("log_history 为空，无法绘图。")
        return

    # 按 step 索引
    step_list = [item["step"] for item in log_history if "step" in item]
    plt.figure(figsize=(8, 5))

    for key in keys:
        xs, ys = [], []
        for item in log_history:
            if key in item:
                xs.append(item.get("step", len(xs)))
                ys.append(item[key])
        if xs and ys:
            plt.plot(xs, ys, label=key)

    plt.xlabel("global_step")
    plt.ylabel("value")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = Path(output_dir) / "loss_curve.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot_loss] 曲线已保存到 {save_path}")


# ------------------------- 2. calculate_tps -----------------------------
def _count_tokens(example) -> int:
    """
    统计单条样本 token 数。默认优先读取 input_ids；
    若数据为字符串则按空格简单 token 化；否则返回 0。
    """
    if isinstance(example, dict):
        # 常见格式：{"input_ids": [...], "attention_mask": [...], ...}
        if "input_ids" in example:
            return len(example["input_ids"])
        # 也可能是 {"prompt": "...", "response": "..."}
        elif "prompt" in example:
            return len(example["prompt"].split())
        elif "text" in example:
            return len(example["text"].split())
    elif isinstance(example, (list, torch.Tensor)):
        return len(example)
    elif isinstance(example, str):
        return len(example.split())
    return 0


def calculate_tps(dataset, metrics: dict, stage: str = "sft") -> float:
    """
    计算有效 tokens per second (TPS)。

    Args:
        dataset: 训练用 Dataset / IterableDataset。
        metrics (dict): trainer.train(...) 返回的 metrics 字典，
                        需要包含 `train_runtime` 或 `runtime`。
        stage (str): 字符串标识，用于从 metrics 中取 runtime 字段，
                     如 stage="sft" 则优先取 "sft_runtime"，否则回退到 "train_runtime"。
    Returns:
        float: tokens/second
    """
    # ---- 1) 统计总 token 数 ----
    total_tokens = 0
    for ex in dataset:
        total_tokens += _count_tokens(ex)

    # ---- 2) 获取耗时 ----
    runtime_keys = [f"{stage}_runtime", "train_runtime", "runtime"]
    runtime = None
    for k in runtime_keys:
        if k in metrics:
            runtime = metrics[k]
            break
    if runtime is None or runtime <= 0:
        print("[calculate_tps] 未找到有效 runtime，返回 0。")
        return 0.0

    tps = total_tokens / runtime
    print(
        f"[calculate_tps] total_tokens={total_tokens:,}, "
        f"runtime={runtime:.2f}s -> TPS={tps:,.1f}"
    )
    return tps
