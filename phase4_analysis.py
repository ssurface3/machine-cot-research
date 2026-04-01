#!/usr/bin/env python3
"""
Phase 4 — Analysis & Figures
==============================
Aggregate evaluation results, compute derived metrics, and produce
publication-quality plots.

Usage:
    python phase4_analysis.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from utils import load_json

# ── Matplotlib defaults ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def gather_results() -> dict:
    """Load all per-level eval files and compute aggregate stats."""
    summary = {}
    for level in config.LEVELS:
        p = config.eval_path(level)
        if not p.exists():
            print(f"  WARNING: Missing {p}, skipping level {level}")
            continue
        records = load_json(p)
        n = len(records)
        acc = sum(r["correct"] for r in records) / n
        tokens = [r["cot_tokens"] for r in records]
        latencies = [r["latency_ms"] for r in records]

        summary[level] = {
            "level": level,
            "name": config.LEVEL_NAMES[level],
            "n": n,
            "accuracy": round(acc, 4),
            "avg_tokens": round(np.mean(tokens), 2),
            "median_tokens": int(np.median(tokens)),
            "std_tokens": round(np.std(tokens), 2),
            "avg_latency_ms": round(np.mean(latencies), 2),
            "median_latency_ms": round(np.median(latencies), 2),
        }
    return summary


def compute_derived(summary: dict) -> dict:
    """Add waste ratio, efficiency score, and per-token accuracy."""
    if 0 not in summary or 4 not in summary:
        print("  WARNING: Need both level 0 and level 4 to compute waste ratio")
    else:
        waste = summary[0]["avg_tokens"] / max(summary[4]["avg_tokens"], 1)
        summary[0]["waste_ratio"] = 1.0
        for level in summary:
            tok = summary[level]["avg_tokens"]
            summary[level]["waste_ratio"] = round(
                summary[0]["avg_tokens"] / max(tok, 1), 2
            )

    for level in summary:
        s = summary[level]
        s["efficiency_score"] = round(
            s["accuracy"] / max(s["avg_tokens"], 1) * 100, 4
        )
    return summary


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_accuracy_vs_tokens(summary: dict):
    """Figure 1: Pareto frontier — accuracy vs average token count."""
    levels = sorted(summary.keys())
    tokens = [summary[l]["avg_tokens"] for l in levels]
    accs = [summary[l]["accuracy"] for l in levels]
    names = [summary[l]["name"] for l in levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tokens, accs, "o-", color="#2563eb", linewidth=2, markersize=9)
    for x, y, name in zip(tokens, accs, names):
        ax.annotate(f"L{levels[names.index(name)]} ({name})",
                     (x, y), textcoords="offset points", xytext=(8, 8),
                     fontsize=10)
    ax.set_xlabel("Average CoT Tokens")
    ax.set_ylabel("Accuracy on GSM8K Test")
    ax.set_title("Pareto Frontier: Accuracy vs. Token Compression")
    ax.grid(True, alpha=0.3)
    out = config.FIGURE_DIR / "pareto_frontier.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_degradation_curve(summary: dict):
    """Figure 2: Accuracy degradation as compression increases."""
    levels = sorted(summary.keys())
    accs = [summary[l]["accuracy"] * 100 for l in levels]
    names = [f"L{l}\n{summary[l]['name']}" for l in levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=["#22c55e", "#84cc16", "#eab308",
                                       "#f97316", "#ef4444"][:len(levels)],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Degradation Across Compression Levels")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    out = config.FIGURE_DIR / "degradation_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_efficiency(summary: dict):
    """Figure 3: Efficiency score (accuracy per token) by level."""
    levels = sorted(summary.keys())
    eff = [summary[l]["efficiency_score"] for l in levels]
    names = [f"L{l}" for l in levels]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, eff, color="#8b5cf6", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Efficiency Score (Accuracy / Tokens × 100)")
    ax.set_title("Efficiency Score by Compression Level")
    ax.grid(axis="y", alpha=0.3)
    out = config.FIGURE_DIR / "efficiency_score.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_latency(summary: dict):
    """Figure 4: Inference latency by level."""
    levels = sorted(summary.keys())
    lat = [summary[l]["avg_latency_ms"] for l in levels]
    names = [f"L{l}" for l in levels]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, lat, color="#06b6d4", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Inference Latency by Compression Level")
    ax.grid(axis="y", alpha=0.3)
    out = config.FIGURE_DIR / "latency.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def print_table(summary: dict):
    """Print a formatted results table (Table 1 for the paper)."""
    levels = sorted(summary.keys())
    header = (f"{'Level':<8} {'Name':<12} {'Accuracy':>10} {'Avg Tok':>10} "
              f"{'Waste×':>8} {'Eff Score':>10} {'Latency ms':>12}")
    print(f"\n{'-'*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")
    for l in levels:
        s = summary[l]
        wr = s.get("waste_ratio", "—")
        print(f"{l:<8} {s['name']:<12} {s['accuracy']:>10.4f} "
              f"{s['avg_tokens']:>10.1f} {str(wr):>8} "
              f"{s['efficiency_score']:>10.4f} {s['avg_latency_ms']:>12.1f}")
    print(f"{'-'*len(header)}\n")


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 4: Analysis & Figures\n")
    summary = gather_results()
    if not summary:
        print("No evaluation data found. Run phase3 first.")
        return

    summary = compute_derived(summary)

    # Save raw results JSON
    from utils import save_json
    save_json(summary, config.RESULTS_PATH)

    print_table(summary)

    # Generate all plots
    plot_accuracy_vs_tokens(summary)
    plot_degradation_curve(summary)
    plot_efficiency(summary)
    plot_latency(summary)

    print("\nPhase 4 complete. Figures saved to:", config.FIGURE_DIR)


if __name__ == "__main__":
    main()
