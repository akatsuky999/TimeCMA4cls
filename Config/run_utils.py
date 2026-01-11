"""
Utility helpers for running TimeCMA experiments on Windows.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path(sys.executable)


def _format_cmd_for_print(cmd: Sequence[str]) -> str:
    """Return a readable shell-style string for logging purposes."""
    parts = []
    for token in cmd:
        if " " in token or "\"" in token:
            parts.append(f"\"{token}\"")
        else:
            parts.append(token)
    return " ".join(parts)


def run_timecma_experiments(
    *,
    experiment_name: str,
    data_subdir: str,
    lr_values: Iterable[float],
    channel_values: Iterable[int] = (32,),    # 原始默认32
    e_layers_values: Iterable[int] = (1,),    # 原始默认1
    n_heads_values: Iterable[int] = (8,),
    d_ff_values: Iterable[int] = (32,),       # 原始默认32
    model_type: Literal['timecma', 'timecma_patch'] = 'timecma',  # 默认用原始结构
    use_embedding: bool = True,
    patch_size: int = 8,                       # 仅 timecma_patch 使用
    stride: int = 8,                           # 仅 timecma_patch 使用
    epochs: int = 50,
    seed: int = 2026,                          # 固定随机种子
    extra_args: Sequence[str] | None = None,
    cuda_device: str | None = None,
) -> None:
    """
    Execute `python src/main.py ...` for TimeCMA model experiments.

    Args:
        experiment_name: Value passed via `--name`.
        data_subdir: Folder inside `TSCMA/datasets`.
        lr_values: Iterable of learning rates.
        channel_values: Iterable of channel dimensions.
        e_layers_values: Iterable of encoder layer counts.
        n_heads_values: Iterable of attention head counts.
        d_ff_values: Iterable of feed-forward dimensions.
        model_type: 'timecma' or 'timecma_patch'.
        use_embedding: Whether to use pre-computed embeddings (required for TimeCMA).
        patch_size: Patch size for patching-based model.
        stride: Stride for patching.
        epochs: Number of training epochs.
        seed: Random seed for reproducibility.
        extra_args: Optional flat list of additional CLI arguments.
        cuda_device: Optional CUDA device index.
    """

    if extra_args is None:
        extra_args = []

    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    
    embedding_dir = './Embeddings'
    d_llm = 768
    print("Using GPT-2 embeddings (dim=768) from ./Embeddings")

    for lr, channel, e_layers, n_heads, d_ff in itertools.product(
        lr_values, channel_values, e_layers_values, n_heads_values, d_ff_values
    ):
        cmd = [
            str(PYTHON_BIN),
            str(PROJECT_ROOT / "src" / "main.py"),
            "--output_dir", "experiments",
            "--comment", f"TimeCMA classification",
            "--name", experiment_name,
            "--records_file", "Classification_records.xls",
            "--data_dir", f"./datasets/{data_subdir}",
            "--data_class", "tsra",
            "--pattern", "TRAIN",
            "--val_pattern", "TEST",
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--optimizer", "RAdam",
            "--task", "classification",
            "--key_metric", "accuracy",
            # TimeCMA 参数
            "--model_type", model_type,
            "--channel", str(channel),
            "--e_layers", str(e_layers),
            "--n_heads", str(n_heads),
            "--d_ff", str(d_ff),
            "--d_llm", str(d_llm),  # GPT-2 embedding dimension
            "--seed", str(seed),
        ]
        
        # patch_size 和 stride 仅在 timecma_patch 模式下使用
        if model_type == 'timecma_patch':
            cmd.extend(["--patch_size", str(patch_size), "--stride", str(stride)])

        if use_embedding:
            cmd.extend(["--use_embedding", "--embedding_dir", embedding_dir])

        cmd.extend(extra_args)

        log_line = _format_cmd_for_print(cmd)
        print(f"\n=== Running TimeCMA: {log_line}\n")

        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
