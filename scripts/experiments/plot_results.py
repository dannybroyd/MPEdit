"""
Aggregate and plot results from MPEdit paper experiments.

Generates all paper figures and tables from saved experiment results:
  - Fig 2: t₀ sweep tradeoff curve (dual Y-axis: success rate vs faithfulness)
  - Table 1: Re-planning comparison table (SDEdit vs Full MPD vs RRT)
  - Fig 4: Sketch gallery (σ × t₀ grid)
  - Fig 5: Inference speed plot (time vs t₀)
  - Fig 6: Diversity analysis plot (pairwise distance vs t₀)

Usage:
  cd scripts/inference
  python ../experiments/plot_results.py --results_dir logs_experiments
"""

import argparse
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(values, key=None):
    """Compute mean from a list of dicts or values, filtering None."""
    if key:
        vals = [v[key] for v in values if v.get(key) is not None]
    else:
        vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _safe_std(values, key=None):
    if key:
        vals = [v[key] for v in values if v.get(key) is not None]
    else:
        vals = [v for v in values if v is not None]
    return float(np.std(vals)) if vals else float("nan")


def _safe_sem(values, key=None):
    """Standard error of the mean."""
    if key:
        vals = [v[key] for v in values if v.get(key) is not None]
    else:
        vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals) / np.sqrt(len(vals)))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: t₀ Sweep Tradeoff Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_t0_sweep(results_dir, save_dir):
    """
    Dual Y-axis plot: success rate (left) vs Fréchet faithfulness (right).
    X-axis: noise level t₀.
    """
    results_path = os.path.join(results_dir, "exp1_t0_sweep", "exp1_results.pt")
    if not os.path.exists(results_path):
        print(f"  Exp1 results not found at {results_path}. Skipping.")
        return
    all_results = torch.load(results_path, weights_only=False)

    noise_levels = sorted(all_results.keys())
    success_rates = []
    success_sems = []
    frechet_means = []
    frechet_sems = []

    for t in noise_levels:
        entries = all_results[t]
        if not entries:
            success_rates.append(float("nan"))
            success_sems.append(0)
            frechet_means.append(float("nan"))
            frechet_sems.append(0)
            continue

        sr_vals = [e["success_rate"] for e in entries if e.get("success_rate") is not None]
        success_rates.append(np.mean(sr_vals) * 100 if sr_vals else float("nan"))
        success_sems.append(np.std(sr_vals) / np.sqrt(len(sr_vals)) * 100 if len(sr_vals) > 1 else 0)

        fr_vals = [e["frechet_to_input_mean"] for e in entries]
        frechet_means.append(np.mean(fr_vals))
        frechet_sems.append(np.std(fr_vals) / np.sqrt(len(fr_vals)) if len(fr_vals) > 1 else 0)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_sr = "#2196F3"
    color_fr = "#FF5722"

    ax1.set_xlabel("Noise Level $t_0$ (DDIM step index)")
    ax1.set_ylabel("Success Rate (%)", color=color_sr)
    line1 = ax1.errorbar(
        noise_levels, success_rates, yerr=success_sems,
        color=color_sr, marker="o", linewidth=2.5, markersize=8,
        capsize=4, label="Success Rate",
    )
    ax1.tick_params(axis="y", labelcolor=color_sr)
    ax1.set_ylim(-5, 105)
    ax1.axhline(y=100, color=color_sr, alpha=0.2, linestyle=":")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Fréchet Distance to Input (faithfulness)", color=color_fr)
    line2 = ax2.errorbar(
        noise_levels, frechet_means, yerr=frechet_sems,
        color=color_fr, marker="s", linewidth=2.5, markersize=8,
        capsize=4, label="Fréchet Distance",
    )
    ax2.tick_params(axis="y", labelcolor=color_fr)

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=12)

    ax1.set_title("$t_0$ Sweep: Success Rate vs Faithfulness Tradeoff")
    ax1.set_xticks(noise_levels)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(save_dir, "fig2_t0_sweep.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Re-planning Comparison
# ─────────────────────────────────────────────────────────────────────────────

def generate_baseline_table(results_dir, save_dir):
    """
    Generate LaTeX table and printed summary comparing SDEdit, Full MPD, RRT.
    """
    results_path = os.path.join(results_dir, "exp2_baselines", "exp2_results.pt")
    if not os.path.exists(results_path):
        print(f"  Exp2 results not found at {results_path}. Skipping.")
        return
    all_results = torch.load(results_path, weights_only=False)

    scenarios = sorted(all_results.keys())
    methods = ["sdedit", "full_mpd", "rrt"]
    method_labels = {"sdedit": "SDEdit", "full_mpd": "Full MPD", "rrt": "RRTConnect"}
    metrics_keys = [
        ("success_rate", "S%", "{:.1%}"),
        ("frechet_to_input_mean", "Fréchet↓", "{:.3f}"),
        ("l2_to_input_mean", "L2↓", "{:.3f}"),
        ("t_inference_total", "Time(s)↓", "{:.3f}"),
        ("path_length_mean", "PathLen", "{:.3f}"),
        ("smoothness_mean", "Smooth", "{:.3f}"),
    ]

    # Print table
    header = f"{'Scenario':>10} | {'Method':>12}"
    for _, label, _ in metrics_keys:
        header += f" | {label:>10}"
    print(f"\n{header}")
    print("-" * len(header))

    # LaTeX output
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Re-planning comparison across difficulty scenarios.}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{ll" + "r" * len(metrics_keys) + "}",
        r"\toprule",
        r"Scenario & Method & " + " & ".join(label for _, label, _ in metrics_keys) + r" \\",
        r"\midrule",
    ]

    for scenario in scenarios:
        for method in methods:
            entries = all_results[scenario][method]
            if not entries:
                continue

            row = f"{scenario:>10} | {method_labels[method]:>12}"
            latex_row = f"{scenario} & {method_labels[method]}"

            for key, _, fmt in metrics_keys:
                val = _safe_mean(entries, key)
                row += f" | {fmt.format(val):>10}"
                latex_row += f" & {fmt.format(val)}"

            print(row)
            latex_row += r" \\"
            latex_lines.append(latex_row)

        print()
        latex_lines.append(r"\midrule")

    latex_lines[-1] = r"\bottomrule"
    latex_lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    # Save LaTeX
    latex_path = os.path.join(save_dir, "table1_baselines.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"  Saved: {latex_path}")

    # Also generate a grouped bar chart
    _plot_baseline_bar_chart(all_results, scenarios, methods, method_labels, save_dir)


def _plot_baseline_bar_chart(all_results, scenarios, methods, method_labels, save_dir):
    """Grouped bar chart comparing methods across scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics_to_plot = [
        ("success_rate", "Success Rate", True),  # (key, label, higher_is_better)
        ("frechet_to_input_mean", "Fréchet Distance", False),
        ("t_inference_total", "Inference Time (s)", False),
    ]

    colors = {"sdedit": "#4CAF50", "full_mpd": "#2196F3", "rrt": "#FF9800"}
    x = np.arange(len(scenarios))
    width = 0.25

    for ax, (key, label, _) in zip(axes, metrics_to_plot):
        for i, method in enumerate(methods):
            vals = []
            errs = []
            for scenario in scenarios:
                entries = all_results[scenario][method]
                vals.append(_safe_mean(entries, key))
                errs.append(_safe_sem(entries, key))

            ax.bar(
                x + i * width, vals, width, yerr=errs,
                label=method_labels[method], color=colors[method],
                capsize=3, alpha=0.85,
            )

        ax.set_xlabel("Scenario")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios)
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Re-planning: SDEdit vs Baselines", fontsize=15, y=1.02)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "fig_baselines_comparison.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Sketch Gallery (Experiment 3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sketch_results(results_dir, save_dir):
    """
    Heatmap / grid showing success rate for each (σ, t₀) combination.
    """
    results_path = os.path.join(results_dir, "exp3_sketch", "exp3_results.pt")
    if not os.path.exists(results_path):
        print(f"  Exp3 results not found at {results_path}. Skipping.")
        return
    all_results = torch.load(results_path, weights_only=False)

    sigmas = sorted(all_results.keys())
    noise_levels = sorted(next(iter(all_results.values())).keys())

    # Build heatmap data
    sr_grid = np.full((len(sigmas), len(noise_levels)), float("nan"))
    fr_grid = np.full((len(sigmas), len(noise_levels)), float("nan"))

    for i, sigma in enumerate(sigmas):
        for j, t_noise in enumerate(noise_levels):
            entries = all_results[sigma][t_noise]
            if entries:
                sr_vals = [e["success_rate"] for e in entries if e.get("success_rate") is not None]
                sr_grid[i, j] = np.mean(sr_vals) * 100 if sr_vals else float("nan")
                fr_vals = [e["frechet_to_input_mean"] for e in entries]
                fr_grid[i, j] = np.mean(fr_vals) if fr_vals else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Success rate heatmap
    im1 = ax1.imshow(sr_grid, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax1.set_xticks(range(len(noise_levels)))
    ax1.set_xticklabels(noise_levels)
    ax1.set_yticks(range(len(sigmas)))
    ax1.set_yticklabels([f"σ={s}" for s in sigmas])
    ax1.set_xlabel("Noise Level $t_0$")
    ax1.set_ylabel("Sketch Noise σ")
    ax1.set_title("Success Rate (%)")
    plt.colorbar(im1, ax=ax1)
    # Annotate cells
    for i in range(len(sigmas)):
        for j in range(len(noise_levels)):
            if not np.isnan(sr_grid[i, j]):
                ax1.text(j, i, f"{sr_grid[i, j]:.0f}", ha="center", va="center", fontsize=9)

    # Faithfulness heatmap
    im2 = ax2.imshow(fr_grid, cmap="RdYlGn_r", aspect="auto")
    ax2.set_xticks(range(len(noise_levels)))
    ax2.set_xticklabels(noise_levels)
    ax2.set_yticks(range(len(sigmas)))
    ax2.set_yticklabels([f"σ={s}" for s in sigmas])
    ax2.set_xlabel("Noise Level $t_0$")
    ax2.set_ylabel("Sketch Noise σ")
    ax2.set_title("Fréchet Distance to Sketch")
    plt.colorbar(im2, ax=ax2)
    for i in range(len(sigmas)):
        for j in range(len(noise_levels)):
            if not np.isnan(fr_grid[i, j]):
                ax2.text(j, i, f"{fr_grid[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.suptitle("Sketch-to-Path: Synthetic Sketch Quality × Noise Level", fontsize=14, y=1.02)
    fig.tight_layout()
    save_path = os.path.join(save_dir, "fig4_sketch_heatmap.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Inference Speed Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed(results_dir, save_dir):
    """
    Inference time vs t₀, with horizontal lines for Full MPD and RRT baselines.
    """
    results_path = os.path.join(results_dir, "exp4_speed", "exp4_results.pt")
    if not os.path.exists(results_path):
        print(f"  Exp4 results not found at {results_path}. Skipping.")
        return
    timing = torch.load(results_path, weights_only=False)

    noise_levels = sorted(timing["sdedit"].keys())
    sdedit_means = [np.mean(timing["sdedit"][t]) for t in noise_levels]
    sdedit_stds = [np.std(timing["sdedit"][t]) for t in noise_levels]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        noise_levels, sdedit_means, yerr=sdedit_stds,
        color="#4CAF50", marker="o", linewidth=2.5, markersize=8,
        capsize=4, label="SDEdit", zorder=5,
    )

    if timing.get("full_mpd"):
        mpd_mean = np.mean(timing["full_mpd"])
        mpd_std = np.std(timing["full_mpd"])
        ax.axhline(y=mpd_mean, color="#2196F3", linewidth=2, linestyle="--", label=f"Full MPD ({mpd_mean:.3f}s)")
        ax.fill_between(
            [noise_levels[0] - 0.5, noise_levels[-1] + 0.5],
            mpd_mean - mpd_std, mpd_mean + mpd_std,
            color="#2196F3", alpha=0.1,
        )

    if timing.get("rrt"):
        rrt_mean = np.mean(timing["rrt"])
        rrt_std = np.std(timing["rrt"])
        ax.axhline(y=rrt_mean, color="#FF9800", linewidth=2, linestyle="-.", label=f"RRTConnect ({rrt_mean:.3f}s)")
        ax.fill_between(
            [noise_levels[0] - 0.5, noise_levels[-1] + 0.5],
            rrt_mean - rrt_std, rrt_mean + rrt_std,
            color="#FF9800", alpha=0.1,
        )

    ax.set_xlabel("Noise Level $t_0$ (DDIM step index)")
    ax.set_ylabel("Inference Time (seconds)")
    ax.set_title("Inference Speed: SDEdit vs Baselines")
    ax.set_xticks(noise_levels)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add speedup annotation
    if timing.get("full_mpd"):
        for t, m in zip(noise_levels, sdedit_means):
            speedup = mpd_mean / m if m > 0 else 0
            if speedup > 1.1:
                ax.annotate(
                    f"{speedup:.1f}×", (t, m),
                    textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=8, color="#666",
                )

    fig.tight_layout()
    save_path = os.path.join(save_dir, "fig5_speed.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Diversity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_diversity(results_dir, save_dir):
    """
    Pairwise distance (diversity) vs t₀.
    """
    results_path = os.path.join(results_dir, "exp5_diversity", "exp5_results.pt")
    if not os.path.exists(results_path):
        print(f"  Exp5 results not found at {results_path}. Skipping.")
        return
    all_results = torch.load(results_path, weights_only=False)

    noise_levels = sorted(all_results.keys())

    div_l2_means = []
    div_l2_sems = []
    div_fr_means = []
    div_fr_sems = []

    for t in noise_levels:
        entries = all_results[t]
        l2_vals = [e["pairwise_diversity_l2"] for e in entries if e.get("pairwise_diversity_l2") is not None]
        fr_vals = [e["pairwise_diversity_frechet"] for e in entries if e.get("pairwise_diversity_frechet") is not None]

        div_l2_means.append(np.mean(l2_vals) if l2_vals else float("nan"))
        div_l2_sems.append(np.std(l2_vals) / np.sqrt(len(l2_vals)) if len(l2_vals) > 1 else 0)
        div_fr_means.append(np.mean(fr_vals) if fr_vals else float("nan"))
        div_fr_sems.append(np.std(fr_vals) / np.sqrt(len(fr_vals)) if len(fr_vals) > 1 else 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        noise_levels, div_l2_means, yerr=div_l2_sems,
        color="#9C27B0", marker="o", linewidth=2.5, markersize=8,
        capsize=4, label="Mean Pairwise L2",
    )
    ax.errorbar(
        noise_levels, div_fr_means, yerr=div_fr_sems,
        color="#FF5722", marker="s", linewidth=2.5, markersize=8,
        capsize=4, label="Mean Pairwise Fréchet",
    )

    ax.set_xlabel("Noise Level $t_0$ (DDIM step index)")
    ax.set_ylabel("Pairwise Distance (diversity)")
    ax.set_title("Output Diversity vs Noise Level")
    ax.set_xticks(noise_levels)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(save_dir, "fig6_diversity.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot MPEdit experiment results")
    parser.add_argument(
        "--results_dir", type=str, default="logs_experiments",
        help="Directory containing experiment results (.pt files)",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save figures (defaults to results_dir/figures)",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Generate only a specific figure: fig2, table1, fig4, fig5, fig6",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.results_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MPEdit Paper — Results Plotting")
    print(f"Results: {args.results_dir}")
    print(f"Figures: {save_dir}")
    print(f"{'='*60}")

    plots = {
        "fig2": ("Fig 2: t₀ Sweep Tradeoff", lambda: plot_t0_sweep(args.results_dir, save_dir)),
        "table1": ("Table 1: Baselines Comparison", lambda: generate_baseline_table(args.results_dir, save_dir)),
        "fig4": ("Fig 4: Sketch Heatmap", lambda: plot_sketch_results(args.results_dir, save_dir)),
        "fig5": ("Fig 5: Inference Speed", lambda: plot_speed(args.results_dir, save_dir)),
        "fig6": ("Fig 6: Diversity Analysis", lambda: plot_diversity(args.results_dir, save_dir)),
    }

    if args.only:
        if args.only in plots:
            name, fn = plots[args.only]
            print(f"\n  Generating {name}...")
            fn()
        else:
            print(f"Unknown figure: {args.only}. Available: {list(plots.keys())}")
            sys.exit(1)
    else:
        for key, (name, fn) in plots.items():
            print(f"\n  Generating {name}...")
            fn()

    print(f"\nDone! Figures saved to {save_dir}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
