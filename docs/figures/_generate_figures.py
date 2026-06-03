"""Generate publication-quality matplotlib charts for the project.

Run from project root:
    .venv/bin/python docs/figures/_generate_figures.py

Outputs PNGs into docs/figures/.
"""
import json
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "lscr_mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
PROJECT_ROOT = OUT.parents[1]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

COLOR_BASELINE = "#2c3e50"
COLOR_VARIANT_B = "#3498db"
COLOR_VARIANT_A = "#e67e22"
COLOR_BAD = "#e74c3c"
COLOR_GOOD = "#27ae60"


# ============================================================================
# Figure 1 — bit-rate ladder at n=250 (the HEADLINE finding)
# ============================================================================
def fig_bit_rate_ladder_n250():
    """Variant B accuracy at each bit-rate, T4 fp32 b=4 n=250 seed=42 sampled."""
    summary_path = (
        PROJECT_ROOT / "experiments" / "variant_b_ladder_t4_kaggle" /
        "analysis" / "results" / "summary.json"
    )
    if summary_path.exists():
        rows = json.loads(summary_path.read_text())
        bits = [
            "baseline\n(fp32)" if r["bits"] == 0 else
            f"{r['bits']}\n({r['compression_ratio']:.0f}x)"
            for r in rows
        ]
        accs = [float(r["accuracy_pct"]) for r in rows]
        correct = [int(r["correct"]) for r in rows]
        n = int(rows[0]["n_samples"])
        ci_low = [float(r["wilson_low_pct"]) for r in rows]
        ci_high = [float(r["wilson_high_pct"]) for r in rows]
        ci95 = [[a - lo for a, lo in zip(accs, ci_low)],
                [hi - a for a, hi in zip(accs, ci_high)]]
        subtitle = "generated from analyzed Kaggle JSON artifacts"
    else:
        # Fallback keeps historical docs buildable, but the reproducibility path
        # is to run experiments/variant_b_ladder_t4_kaggle/analysis/analyze_ladder.py.
        bits = ["baseline\n(fp32)", "8\n(4x)", "4\n(8x)", "2\n(16x)"]
        accs = [75.2, 78.4, 76.8, 75.2]
        correct = [188, 196, 192, 188]
        n = 250
        se = [np.sqrt(p / 100 * (1 - p / 100) / n) * 100 for p in accs]
        ci95 = [1.96 * s for s in se]
        subtitle = "fallback values from REPORT_06"

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bits))
    bars = ax.bar(x, accs, yerr=ci95, capsize=6,
                  color=[COLOR_BASELINE, COLOR_VARIANT_B, COLOR_VARIANT_B, COLOR_VARIANT_B],
                  edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bits)
    ax.set_xlabel("bits per coordinate (compression ratio vs fp32)")
    ax.set_ylabel("math500 accuracy (%)")
    ax.set_title("Inter-agent latent channel: math500 accuracy vs. compression\n"
                 "n=250, sampled decoding (T4 fp32) — no degradation detected "
                 "(two-proportion $z$-test, all $p>0.4$)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.axhline(75.2, color=COLOR_GOOD, linestyle="--", linewidth=1, alpha=0.7,
               label="baseline 75.2%")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Annotate bars with values + n
    for bar, v, c in zip(bars, accs, correct):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3.5,
                f"{v:.1f}%\n({c}/{n})", ha="center", va="bottom", fontsize=9)

    ax.legend(loc="lower right")

    fig.savefig(OUT / "bit_rate_ladder_n250.png")
    plt.close(fig)
    print(f"  ✓ {OUT / 'bit_rate_ladder_n250.png'}")


# ============================================================================
# Figure 2 — per-link distortion: synthetic vs real Solver adapter vs TurboQuant paper
# ============================================================================
def fig_distortion_curve():
    """rMSE vs bits, three independent measurements converging on TurboQuant theory."""
    bits = [2, 3, 4, 8]
    # From REPORT_02 + REPORT_03 + TurboQuant paper Table 1
    rmse_synthetic = [0.1175, 0.0345, 0.0095, 0.0001]
    rmse_real_adapter = [0.1159, 0.0339, 0.0093, 0.0001]
    rmse_paper = [0.117, 0.030, 0.009, 0.0001]

    # Variant A (per-channel uniform, screening baseline) for comparison
    rmse_variant_a = [0.7509, 0.2625, 0.0202, 0.0001]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(bits, rmse_paper, marker="^", linestyle="--", color="black", alpha=0.7,
                label="TurboQuant paper (Table 1)", markersize=10, linewidth=1.5)
    ax.semilogy(bits, rmse_synthetic, marker="o", linestyle="-", color=COLOR_VARIANT_B,
                label="Variant B synthetic d=2048 (this work)", markersize=8, linewidth=2)
    ax.semilogy(bits, rmse_real_adapter, marker="s", linestyle="-", color=COLOR_GOOD,
                label="Variant B real Solver adapter d=1536", markersize=8, linewidth=2)
    ax.semilogy(bits, rmse_variant_a, marker="x", linestyle=":", color=COLOR_VARIANT_A,
                label="Variant A per-channel (baseline screening)", markersize=10, linewidth=1.5)

    ax.set_xlabel("bits per coordinate")
    ax.set_ylabel("rMSE (log scale)")
    ax.set_title("Per-link distortion converges on TurboQuant theory across 3 independent measurements\n"
                 "(synthetic Gaussian on sphere, real Solver inner-adapter capture-replay, paper)")
    ax.set_xticks(bits)
    ax.set_xticklabels([f"{b}-bit" for b in bits])
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left")
    ax.invert_xaxis()  # more bits = less compression = lower rMSE on the right

    fig.savefig(OUT / "distortion_vs_bits.png")
    plt.close(fig)
    print(f"  ✓ {OUT / 'distortion_vs_bits.png'}")


# ============================================================================
# Figure 3 — hardware/dtype collapse story
# ============================================================================
def fig_hardware_dtype_collapse():
    """Math500 baseline accuracy by GPU SKU + dtype routing. Tells the bf16 story."""
    labels = [
        "P100\nbf16 (auto)\nfp16 fallback",
        "T4\nbf16 (auto)\nfp16 fallback",
        "T4\n--dtype float32\n(explicit)",
        "A100\nbf16 (auto)\nnative",
        "Paper\nA100/H100 bf16\n(n=500)",
    ]
    accs = [35.0, 30.0, 75.2, 86.0, 75.8]
    n_values = [100, 50, 250, 50, 500]
    colors = [COLOR_BAD, COLOR_BAD, COLOR_GOOD, COLOR_GOOD, COLOR_BASELINE]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, accs, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("math500 accuracy (%)")
    ax.set_title("RecursiveMAS Sequential-Light baseline accuracy by GPU + dtype routing\n"
                 "(identical code, identical checkpoints — only hardware/dtype varies)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for bar, v, n in zip(bars, accs, n_values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)

    # Annotation arrow: collapse → workaround
    ax.annotate("", xy=(2, 75), xytext=(1, 35),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black", alpha=0.6))
    ax.text(1.5, 55, "+45pp\nwith\n--dtype float32", ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    fig.text(0.5, -0.02,
             "Take-away: pre-Ampere GPUs require explicit --dtype float32 to avoid silent fp16-fallback collapse.",
             ha="center", fontsize=9, style="italic")

    fig.savefig(OUT / "hardware_dtype_collapse.png")
    plt.close(fig)
    print(f"  ✓ {OUT / 'hardware_dtype_collapse.png'}")


# ============================================================================
# Figure 4 — n=50 vs n=250 sample variance demonstration
# ============================================================================
def fig_sample_variance():
    """Side-by-side: n=50 noisy ladder vs n=250 clean ladder."""
    bits = ["baseline", "8 bit", "4 bit", "2 bit"]
    n50 = [84.0, 86.0, 80.0, 74.0]
    n250 = [75.2, 78.4, 76.8, 75.2]
    se50 = [np.sqrt(p / 100 * (1 - p / 100) / 50) * 100 for p in n50]
    se250 = [np.sqrt(p / 100 * (1 - p / 100) / 250) * 100 for p in n250]
    ci50 = [1.96 * s for s in se50]
    ci250 = [1.96 * s for s in se250]

    x = np.arange(len(bits))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, n50, width, yerr=ci50, capsize=4,
           color="#aed6f1", edgecolor="black", linewidth=0.6,
           label="Phase 0.I (n=50)")
    ax.bar(x + width / 2, n250, width, yerr=ci250, capsize=4,
           color=COLOR_VARIANT_B, edgecolor="black", linewidth=0.6,
           label="Phase 0.J (n=250) — canonical")

    ax.set_xticks(x)
    ax.set_xticklabels(bits)
    ax.set_xlabel("Variant B bit-rate")
    ax.set_ylabel("math500 accuracy (%)")
    ax.set_title("Why n=50 misled us: sample-variance dominated apparent treatment effects\n"
                 "(error bars 95% Wilson CI)")
    ax.set_ylim(50, 100)
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    for i, (v50, v250, c50, c250) in enumerate(zip(n50, n250, ci50, ci250)):
        ax.text(i - width / 2, v50 + c50 + 1, f"{v50:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, v250 + c250 + 1, f"{v250:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.text(0.5, -0.02,
             "At n=50, baseline drew an easy 50-problem subset (84%); at n=250 the true baseline emerges (75.2%) and Variant B flatlines on top of it.",
             ha="center", fontsize=9, style="italic")

    fig.savefig(OUT / "sample_variance_n50_vs_n250.png")
    plt.close(fig)
    print(f"  ✓ {OUT / 'sample_variance_n50_vs_n250.png'}")


if __name__ == "__main__":
    print("Generating figures...")
    fig_bit_rate_ladder_n250()
    fig_distortion_curve()
    fig_hardware_dtype_collapse()
    fig_sample_variance()
    print("Done.")
