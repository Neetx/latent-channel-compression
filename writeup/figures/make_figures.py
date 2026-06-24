#!/usr/bin/env python3
"""Generate the paper's vector figures from the committed result summaries.

Design follows the figure conventions in the ml-paper-writing guide: vector (PDF) output,
no title inside the figure (the LaTeX caption serves that), an Okabe-Ito colorblind-safe
palette, hatching so the bars are also distinguishable in grayscale, and self-contained
content. Run: python make_figures.py  (writes dissociation.pdf, tier_task_divergence.pdf)

Values are the canonical committed numbers:
  - accuracy delta / churn  -> results/.../SUMMARY.md (compare_cells)
  - divergence within 128    -> results/tier2_logit_fidelity_SUMMARY.md
  - per-rotation divergence  -> results/rotation_matrix_SUMMARY.md
"""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-fig")
Path("/tmp/mpl-fig").mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9, "legend.frameon": False, "legend.fontsize": 8,
})
# Okabe-Ito colorblind-safe palette
OI = {"green": "#009E73", "orange": "#E69F00", "blue": "#0072B2",
      "sky": "#56B4E9", "vermillion": "#D55E00", "grey": "#999999"}

# ----------------------------------------------------------------------------
# Figure 1 — the dissociation: aggregate accuracy is preserved, trajectories are not.
# Four non-confounded cells; per cell three quantities, all "% of problems affected".
# ----------------------------------------------------------------------------
cells = ["Math500\nlight", "Math500\nscaled", "MBPP+\nlight", "MBPP+\nscaled"]
abs_delta = [2.0, 2.4, 0.0, 2.0]    # |paired greedy accuracy delta| (pp)
churn = [10.0, 8.8, 9.6, 4.4]        # % of problems whose correctness flips
diverg = [86.4, 80.4, 92.8, 51.2]    # % of greedy trajectories diverging within 128

x = np.arange(len(cells))
w = 0.26
fig, ax = plt.subplots(figsize=(5.6, 3.0))
b1 = ax.bar(x - w, abs_delta, w, label=r"$|\Delta$ accuracy$|$ (pp)", color=OI["green"], hatch="//")
b2 = ax.bar(x, churn, w, label="answer churn", color=OI["orange"], hatch="..")
b3 = ax.bar(x + w, diverg, w, label="trajectory divergence", color=OI["blue"], hatch="xx")
for bars in (b1, b2, b3):
    for r in bars:
        ax.annotate(f"{r.get_height():.0f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=6.5, xytext=(0, 1), textcoords="offset points")
ax.set_xticks(x)
ax.set_xticklabels(cells)
ax.set_ylabel("% of problems affected")
ax.set_ylim(0, 100)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13), columnspacing=1.0, handlelength=1.4)
fig.savefig(OUT / "dissociation.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 2 — the tier x task contrast, with quantizer-rotation variability.
# MBPP+ tiers averaged over 5 rotations (error bar = std across rotations);
# Math500 tiers are the single seed-42 rotation (no error bar).
# ----------------------------------------------------------------------------
mbpp_light = np.array([92.8, 91.6, 93.2, 92.8, 93.6])   # divergence within 128, 5 rotations
mbpp_scaled = np.array([51.2, 53.6, 52.4, 50.4, 55.2])
math_light, math_scaled = 86.4, 80.4                     # seed 42 only

groups = ["MBPP+ (code)", "Math500 (math)"]
xg = np.arange(len(groups))
wg = 0.34
light_vals = [mbpp_light.mean(), math_light]
scaled_vals = [mbpp_scaled.mean(), math_scaled]
light_err = [mbpp_light.std(), 0.0]
scaled_err = [mbpp_scaled.std(), 0.0]

fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.bar(xg - wg / 2, light_vals, wg, yerr=light_err, capsize=4, label="light (~1.5B)",
       color=OI["sky"], hatch="//", error_kw=dict(ecolor="black", lw=1))
ax.bar(xg + wg / 2, scaled_vals, wg, yerr=scaled_err, capsize=4, label="scaled (~4B)",
       color=OI["vermillion"], hatch="xx", error_kw=dict(ecolor="black", lw=1))
for xi, lv, sv in zip(xg, light_vals, scaled_vals):
    ax.annotate(f"{lv:.1f}", (xi - wg / 2, lv), ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
    ax.annotate(f"{sv:.1f}", (xi + wg / 2, sv), ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(xg)
ax.set_xticklabels(groups)
ax.set_ylabel("trajectory divergence within 128 (%)")
ax.set_ylim(0, 105)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13), columnspacing=1.4, handlelength=1.4)
fig.savefig(OUT / "tier_task_divergence.pdf")
plt.close(fig)

# ----------------------------------------------------------------------------
# Figure 3 — independent cloud (T4 fp32) Math500 sampled ladder. Honest Wilson 95%
# confidence intervals, no in-figure title, and no significance-test annotation.
# ----------------------------------------------------------------------------
import math
def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p * 100, (p - (centre - half)) * 100, ((centre + half) - p) * 100

counts, n_cloud = [188, 196, 192, 188], 250          # T4 fp32 Math500, of 250
bits_lab = ["baseline\n(fp32)", "8\n(4x)", "4\n(8x)", "2\n(16x)"]
vals = [c / n_cloud * 100 for c in counts]
errs = np.array([wilson(c, n_cloud)[1:] for c in counts]).T
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.bar(range(4), vals, 0.6, yerr=errs, capsize=4,
       color=[OI["grey"]] + [OI["blue"]] * 3, error_kw=dict(ecolor="black", lw=1))
ax.axhline(vals[0], ls="--", color=OI["green"], lw=1, label=f"baseline {vals[0]:.1f}%")
for i, v in enumerate(vals):
    ax.annotate(f"{v:.1f}", (i, v + errs[1][i]), ha="center", va="bottom",
                fontsize=7, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(range(4))
ax.set_xticklabels(bits_lab)
ax.set_ylabel("Math500 accuracy (%)")
ax.set_xlabel("bits per coordinate (compression vs fp32)")
ax.set_ylim(0, 100)
ax.legend(loc="lower center")
fig.savefig(OUT / "cloud_ladder.pdf")
plt.close(fig)

print("wrote:", OUT / "dissociation.pdf", OUT / "tier_task_divergence.pdf", OUT / "cloud_ladder.pdf")
print(f"  MBPP+ rotation means: light={mbpp_light.mean():.1f}+-{mbpp_light.std():.1f}  "
      f"scaled={mbpp_scaled.mean():.1f}+-{mbpp_scaled.std():.1f}")
