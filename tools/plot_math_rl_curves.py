#!/usr/bin/env python3
"""Plot publication-style curves from a slime Math RL log."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


TRAIN_RE = re.compile(r"model\.py:\d+ - step (\d+): (\{.*\})")
EVAL_RE = re.compile(r"rollout_metrics\.py:\d+ - eval (\d+): (\{.*\})")
ROLLOUT_RE = re.compile(r"rollout_metrics\.py:\d+ - perf (\d+): (\{.*\})")


def parse_records(path: Path, pattern: re.Pattern[str]) -> list[tuple[int, dict]]:
    records = []
    for match in pattern.finditer(path.read_text(errors="replace")):
        try:
            records.append((int(match.group(1)), ast.literal_eval(match.group(2))))
        except (SyntaxError, ValueError):
            continue
    return records


def moving_average(values: list[float], window: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return array
    window = min(window, len(array))
    kernel = np.ones(window) / window
    padded = np.pad(array, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def ema(values: list[float], alpha: float = 0.25) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return array
    output = np.empty_like(array)
    output[0] = array[0]
    for index in range(1, len(array)):
        output[index] = alpha * array[index] + (1 - alpha) * output[index - 1]
    return output


def get_values(records: list[tuple[int, dict]], key: str) -> tuple[list[int], list[float]]:
    points = [(step, float(metrics[key])) for step, metrics in records if key in metrics]
    if not points:
        return [], []
    steps, values = zip(*points)
    return list(steps), list(values)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
        }
    )


def plot(log_path: Path, output_prefix: Path) -> None:
    train = parse_records(log_path, TRAIN_RE)
    evaluation = parse_records(log_path, EVAL_RE)
    rollout = parse_records(log_path, ROLLOUT_RE)
    if not train:
        raise RuntimeError(f"No train metrics found in {log_path}")

    style()
    colors = {"blue": "#2455A4", "orange": "#D97904", "green": "#2A7F62", "red": "#B23A48"}
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.05), constrained_layout=True)

    # Panel A: eval accuracy. Keep raw points visible because eval is only avg@2.
    ax = axes[0, 0]
    for key, label, color in [
        ("eval/aime2024", "AIME 2024", colors["blue"]),
        ("eval/aime2025", "AIME 2025", colors["orange"]),
    ]:
        x, y = get_values(evaluation, key)
        if x:
            ax.plot(x, y, "o", ms=3.2, alpha=0.34, color=color)
            ax.plot(x, moving_average(y, 3), lw=2.0, color=color, label=label)
    ax.set_title("Evaluation accuracy (avg@2)", loc="left", fontweight="bold")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.2, 0.7)
    ax.legend(frameon=False, ncol=2, loc="lower right")

    # Panel B: train / rollout mismatch.
    ax = axes[0, 1]
    x, y = get_values(train, "train/train_rollout_logprob_abs_diff")
    ax.plot(x, y, "o", ms=2.8, alpha=0.28, color=colors["red"])
    ax.plot(x, ema(y, 0.22), lw=2.0, color=colors["red"], label="EMA")
    ax.set_title("Train–rollout logprob mismatch", loc="left", fontweight="bold")
    ax.set_xlabel("Train step")
    ax.set_ylabel("Mean absolute difference")
    ax.legend(frameon=False, loc="upper right")

    # Panel C: useful GRPO groups. Zero-std counts are measured among 32 groups.
    ax = axes[1, 0]
    x, positive = get_values(rollout, "rollout/zero_std/count_1.0")
    _, negative = get_values(rollout, "rollout/zero_std/count_-1.0")
    if positive and negative:
        informative = [32 - p - n for p, n in zip(positive, negative, strict=True)]
        ax.plot(x, informative, "o", ms=2.8, alpha=0.28, color=colors["green"])
        ax.plot(x, moving_average(informative, 5), lw=2.0, color=colors["green"], label="smoothed")
    ax.axhline(32, color="#777777", lw=0.8, ls="--", label="all 32 groups")
    ax.set_title("Groups with non-zero reward variance", loc="left", fontweight="bold")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Groups / 32")
    ax.set_ylim(0, 34)
    ax.legend(frameon=False, loc="lower right")

    # Panel D: truncation, train versus eval.
    ax = axes[1, 1]
    x, y = get_values(rollout, "rollout/truncated_ratio")
    ax.plot(x, y, "o", ms=2.8, alpha=0.24, color=colors["blue"])
    ax.plot(x, moving_average(y, 5), lw=2.0, color=colors["blue"], label="train rollout")
    x_eval, y_eval = get_values(evaluation, "eval/aime2024-truncated_ratio")
    if x_eval:
        ax.plot(x_eval, y_eval, "o", ms=2.8, alpha=0.28, color=colors["orange"])
        ax.plot(x_eval, moving_average(y_eval, 3), lw=2.0, color=colors["orange"], label="AIME 2024 eval")
    ax.set_title("Response truncation", loc="left", fontweight="bold")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("YuLan MoE Math RL — training dynamics", fontsize=11, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.005, f"Source: {log_path.name}; points are raw, lines are light smoothing.", fontsize=7.5, color="#596273")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"))
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    plot(args.log, args.output_prefix)


if __name__ == "__main__":
    main()
