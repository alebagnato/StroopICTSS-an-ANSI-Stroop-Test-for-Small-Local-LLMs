#!/usr/bin/env python3
"""
stroop_plot.py — ANSI Stroop test results visualization
Generates 3 PNG figures from the CSV produced by stroop_ansi_llm.py

Dependencies: matplotlib (pip install matplotlib)
Usage       : python3 stroop_plot.py stroop_results_20260625_114157.csv
"""

import csv
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "gemma":   "#378ADD",
    "llama":   "#639922",
    "mistral": "#D85A30",
}
LINESTYLES = {
    "gemma":   "-",
    "llama":   "--",
    "mistral": ":",
}
LENGTHS    = [1, 5, 10, 20]
CONDITIONS = ["congruent", "incongruent", "mixed"]
MODELS     = ["gemma", "llama", "mistral"]

# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["length"]           = int(r["length"])
        r["accuracy"]         = float(r["accuracy"])
        r["word_read_errors"] = int(r["word_read_errors"])
        r["elapsed_s"]        = float(r["elapsed_s"])
    return rows

def get(data, model, length, condition, field="accuracy"):
    row = next((r for r in data
                if r["model_key"] == model
                and r["length"]   == length
                and r["condition"] == condition), None)
    return row[field] if row else None

def is_parse_failed(data, model, length, condition):
    row = next((r for r in data
                if r["model_key"] == model
                and r["length"]   == length
                and r["condition"] == condition), None)
    if row is None:
        return False
    val = row.get("parse_failed", False)
    return str(val).strip().lower() == "true"

# ── Global style ──────────────────────────────────────────────────────────────

def apply_style(ax, title, ylabel, ylim=(0, 110)):
    ax.set_title(title, fontsize=12, fontweight="normal", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(LENGTHS)
    ax.set_xticklabels([f"n={n}" for n in LENGTHS])
    ax.set_ylim(*ylim)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.grid(axis="y", which="minor", color="#f0f0f0", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

def legend(ax):
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=COLORS[m], linestyle=LINESTYLES[m],
                       linewidth=2, marker="o", markersize=5, label=m)
            for m in MODELS
        ],
        fontsize=9, framealpha=0.6, edgecolor="#cccccc"
    )

# ── Figure 1 — Accuracy by condition ─────────────────────────────────────────

def fig_accuracy(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle("ANSI Stroop Test — Accuracy (%)", fontsize=13, y=1.02)

    for ax, condition in zip(axes, CONDITIONS):
        for model in MODELS:
            vals = [get(data, model, n, condition) for n in LENGTHS]
            ax.plot(LENGTHS, vals,
                    color=COLORS[model], linestyle=LINESTYLES[model],
                    linewidth=2, marker="o", markersize=6, label=model)
            for n, v in zip(LENGTHS, vals):
                if v is not None:
                    ax.annotate(f"{v:.0f}%", (n, v),
                                textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=8,
                                color=COLORS[model])
        apply_style(ax, condition, "accuracy (%)" if condition == "congruent" else "")
        legend(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "stroop_1_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

# ── Figure 2 — Interference (word_read_errors) ───────────────────────────────

def fig_interference(data, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Stroop Interference (model names the word instead of the ANSI color)",
                 fontsize=13, y=1.02)

    for ax, condition in zip(axes, ["incongruent", "mixed"]):
        x = range(len(LENGTHS))
        width = 0.25
        for i, model in enumerate(MODELS):
            vals = [get(data, model, n, condition, "word_read_errors") for n in LENGTHS]
            offset = (i - 1) * width
            bars = ax.bar([xi + offset for xi in x], vals,
                          width=width * 0.9,
                          color=COLORS[model], alpha=0.85,
                          label=model)
            for bar, v in zip(bars, vals):
                if v:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            str(v), ha="center", va="bottom",
                            fontsize=8, color=COLORS[model])

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"n={n}" for n in LENGTHS])
        ax.set_title(f"condition: {condition}", fontsize=11)
        ax.set_ylabel("interference count" if condition == "incongruent" else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
        legend(ax)

    fig.tight_layout()
    path = os.path.join(out_dir, "stroop_2_interference.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

# ── Figure 3 — Cognitive signatures (heatmap) ────────────────────────────────

def fig_heatmap(data, out_dir):
    """
    Heatmap: model × (condition × length) — accuracy as color.
    Quick read of cognitive signatures across architectures.

    Cells with parse_failed=True are hatched to distinguish format
    failures (no valid response obtained) from genuine cognitive
    errors (valid response, wrong answer).
    """
    col_labels = [f"{cond[:3]}\nn={n}" for cond in CONDITIONS for n in LENGTHS]
    matrix      = []
    fail_matrix = []   # True where parse_failed, for hatching
    for model in MODELS:
        row      = []
        fail_row = []
        for condition in CONDITIONS:
            for n in LENGTHS:
                v = get(data, model, n, condition)
                row.append(v if v is not None else 0)
                fail_row.append(is_parse_failed(data, model, n, condition))
        matrix.append(row)
        fail_matrix.append(fail_row)

    matrix = [[v / 100 for v in row] for row in matrix]

    fig, ax = plt.subplots(figsize=(11, 3))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)
    ax.set_title("LLMs Cognitive Signatures — Accuracy (red=0%, green=100%)",
                 fontsize=12, pad=10)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if fail_matrix[i][j]:
                # Overlay light blue for parse failures — distinct from the
                # red/green accuracy scale, keeps text legible
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#AED6F1", edgecolor="white", linewidth=1.5
                ))
                ax.text(j, i, "N/A*",
                        ha="center", va="center", fontsize=9, color="#1B4F72")
            else:
                ax.text(j, i, f"{val*100:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if val < 0.45 or val > 0.75 else "#333")

    # Separators between conditions
    for x in [2.5, 5.5]:
        ax.axvline(x, color="white", linewidth=2)

    # Condition labels on top
    for idx, cond in enumerate(CONDITIONS):
        center = idx * 3 + 1
        ax.text(center, -0.7, cond, ha="center", va="center",
                fontsize=9, color="#555",
                transform=ax.get_xaxis_transform())

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # Caption note for light-blue cells, only if any exist
    if any(any(row) for row in fail_matrix):
        fig.text(0.01, -0.08,
                  "N/A* (light blue) = parse failure, no valid JSON response obtained — not a cognitive error",
                  fontsize=8, color="#555", ha="left")

    fig.tight_layout()
    path = os.path.join(out_dir, "stroop_3_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "stroop_results.csv"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or "."

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    data = load_csv(csv_path)
    print(f"\nANSI Stroop — {len(data)} rows loaded from {csv_path}")
    print(f"Figures saved to: {out_dir}\n")

    fig_accuracy(data, out_dir)
    fig_interference(data, out_dir)
    fig_heatmap(data, out_dir)

    print("\nDone.")
