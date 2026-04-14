"""
Generate publication-ready figures for NCUR binning paper.
All data from 1000-replicate bootstrap analysis.
Outputs: 300 DPI PNG files.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import shutil

# ─── Output directory ───────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Data from bootstrap log (1000 replicates, 95% CIs) ────────────────────
methods = [
    "EqualFreqChi",
    "ChiMerge",
    "DecisionTree",
    "MAPA",
    "ChiIsotonic",
    "IsotonicRegression",
    "MultiIntervalDisc.",
    "ConditionalInference",
    "EqualFreq",
    "IsotonicChi",
]

auroc_mean = [0.8850, 0.8842, 0.8832, 0.8816, 0.8805, 0.8801, 0.8793, 0.8765, 0.8764, 0.8745]
auroc_lo   = [0.8809, 0.8793, 0.8786, 0.8774, 0.8754, 0.8754, 0.8625, 0.8708, 0.8700, 0.8542]
auroc_hi   = [0.8887, 0.8889, 0.8876, 0.8857, 0.8854, 0.8844, 0.8851, 0.8819, 0.8832, 0.8806]

ks_mean = [0.6135, 0.6098, 0.6136, 0.6076, 0.6046, 0.6135, 0.6110, 0.5982, 0.5961, 0.6038]
ks_lo   = [0.6037, 0.5958, 0.6022, 0.5970, 0.5926, 0.6013, 0.5849, 0.5852, 0.5772, 0.5637]
ks_hi   = [0.6234, 0.6225, 0.6248, 0.6187, 0.6173, 0.6246, 0.6229, 0.6150, 0.6171, 0.6193]

# Sort by AUROC for forest plots (already sorted above)
auroc_ci_width = [hi - lo for hi, lo in zip(auroc_hi, auroc_lo)]
ks_ci_width    = [hi - lo for hi, lo in zip(ks_hi, ks_lo)]

# ─── Categorize methods ─────────────────────────────────────────────────────
METHOD_TYPE = {
    "EqualFreq":           "Unsupervised",
    "DecisionTree":        "Supervised",
    "ChiMerge":            "Supervised",
    "MAPA":                "Supervised",
    "ConditionalInference":"Supervised",
    "MultiIntervalDisc.":  "Supervised",
    "IsotonicRegression":  "Supervised",
    "ChiIsotonic":         "Hybrid",
    "IsotonicChi":         "Hybrid",
    "EqualFreqChi":        "Hybrid",
}

TYPE_COLORS = {
    "Unsupervised": "#5B8DB8",   # Steel blue
    "Supervised":   "#E07B54",   # Burnt orange
    "Hybrid":       "#6BAF6B",   # Forest green
}

def get_colors(method_list):
    return [TYPE_COLORS[METHOD_TYPE[m]] for m in method_list]


# ─── Global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — AUROC Forest Plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig_auroc_forest():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    y_pos = np.arange(len(methods))
    colors = get_colors(methods)
    xerr_lo = [m - lo for m, lo in zip(auroc_mean, auroc_lo)]
    xerr_hi = [hi - m for m, hi in zip(auroc_mean, auroc_hi)]

    ax.errorbar(auroc_mean, y_pos, xerr=[xerr_lo, xerr_hi],
                fmt='none', ecolor='#555555', elinewidth=1.2, capsize=4, capthick=1.0, zorder=2)
    ax.scatter(auroc_mean, y_pos, c=colors, s=70, zorder=3, edgecolors='white', linewidths=0.6)

    median_auroc = np.median(auroc_mean)
    ax.axvline(median_auroc, color='#AAAAAA', linestyle='--', linewidth=0.8, zorder=1)
    ax.text(median_auroc, len(methods) - 0.3, f'Median: {median_auroc:.4f}',
            ha='center', va='bottom', fontsize=9, color='#888888')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11)
    ax.set_xlabel("AUROC", fontsize=13)
    ax.set_title("AUROC with 95% Empirical Confidence Intervals", fontsize=14, fontweight='bold', pad=12)
    ax.invert_yaxis()

    # Add CI text annotations
    for i in range(len(methods)):
        ax.text(auroc_hi[i] + 0.00055, y_pos[i], f'{auroc_mean[i]:.4f}',
                va='center', fontsize=9, color='#444444')

    # Legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    ax.legend(handles=patches, loc='upper left', fontsize=10, framealpha=0.95)

    ax.set_xlim(0.850, 0.895)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_auroc_forest.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — KS Forest Plot
# ═══════════════════════════════════════════════════════════════════════════════
def fig_ks_forest():
    # Sort by KS mean
    idx = np.argsort(ks_mean)[::-1]
    m_sorted = [methods[i] for i in idx]
    ks_m = [ks_mean[i] for i in idx]
    ks_l = [ks_lo[i] for i in idx]
    ks_h = [ks_hi[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    y_pos = np.arange(len(m_sorted))
    colors = get_colors(m_sorted)
    xerr_lo = [m - lo for m, lo in zip(ks_m, ks_l)]
    xerr_hi = [hi - m for m, hi in zip(ks_m, ks_h)]

    ax.errorbar(ks_m, y_pos, xerr=[xerr_lo, xerr_hi],
                fmt='none', ecolor='#555555', elinewidth=1.2, capsize=4, capthick=1.0, zorder=2)
    ax.scatter(ks_m, y_pos, c=colors, s=70, zorder=3, edgecolors='white', linewidths=0.6)

    median_ks = np.median(ks_m)
    ax.axvline(median_ks, color='#AAAAAA', linestyle='--', linewidth=0.8, zorder=1)
    ax.text(median_ks, len(m_sorted) - 0.3, f'Median: {median_ks:.4f}',
            ha='center', va='bottom', fontsize=9, color='#888888')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(m_sorted, fontsize=11)
    ax.set_xlabel("KS Statistic", fontsize=13)
    ax.set_title("KS Statistic with 95% Empirical Confidence Intervals", fontsize=14, fontweight='bold', pad=12)
    ax.invert_yaxis()

    for i in range(len(m_sorted)):
        ax.text(ks_h[i] + 0.0006, y_pos[i], f'{ks_m[i]:.4f}',
                va='center', fontsize=9, color='#444444')

    patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    ax.legend(handles=patches, loc='upper left', fontsize=10, framealpha=0.95)

    ax.set_xlim(0.555, 0.640)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_ks_forest.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CI Width Bar Chart (Stability)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_ci_width():
    # Sort by AUROC CI width (narrowest first)
    idx = np.argsort(auroc_ci_width)
    m_sorted  = [methods[i] for i in idx]
    aw_sorted = [auroc_ci_width[i] for i in idx]
    kw_sorted = [ks_ci_width[i] for i in idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.9), sharey=True)

    y_pos = np.arange(len(m_sorted))
    colors = get_colors(m_sorted)

    # AUROC CI width
    bars1 = ax1.barh(y_pos, aw_sorted, color=colors, edgecolor='white', linewidth=0.5, height=0.65)
    ax1.set_xlabel("AUROC CI Width (97.5th − 2.5th)", fontsize=11)
    ax1.set_title("AUROC Stability", fontsize=13, fontweight='bold')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(m_sorted, fontsize=11)
    for i, v in enumerate(aw_sorted):
        ax1.text(v + 0.00035, y_pos[i], f'{v:.4f}', va='center', fontsize=9, color='#444444')

    # KS CI width (sorted same order)
    bars2 = ax2.barh(y_pos, kw_sorted, color=colors, edgecolor='white', linewidth=0.5, height=0.65)
    ax2.set_xlabel("KS CI Width (97.5th − 2.5th)", fontsize=11)
    ax2.set_title("KS Stability", fontsize=13, fontweight='bold')
    for i, v in enumerate(kw_sorted):
        ax2.text(v + 0.0006, y_pos[i], f'{v:.4f}', va='center', fontsize=9, color='#444444')

    fig.suptitle("Confidence Interval Width — Narrower = More Stable", fontsize=14, fontweight='bold', y=1.02)
    patches = [mpatches.Patch(color=c, label=t) for t, c in TYPE_COLORS.items()]
    ax2.legend(handles=patches, loc='lower right', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_ci_width_stability.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — AUROC vs. KS Scatter with 2D Error Bars  (numbered legend)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_auroc_vs_ks():
    fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(10.8, 6.4),
                                      gridspec_kw={'width_ratios': [3, 1]})
    colors = get_colors(methods)

    # Simple digit labels 1-10
    labels = [str(i+1) for i in range(len(methods))]

    for i, m in enumerate(methods):
        xerr = [[auroc_mean[i] - auroc_lo[i]], [auroc_hi[i] - auroc_mean[i]]]
        yerr = [[ks_mean[i] - ks_lo[i]], [ks_hi[i] - ks_mean[i]]]
        ax.errorbar(auroc_mean[i], ks_mean[i], xerr=xerr, yerr=yerr,
                    fmt='none', ecolor='#AAAAAA',
                    elinewidth=1.0, capsize=3.5, capthick=1.0, zorder=2)
        # Colored circle marker
        ax.plot(auroc_mean[i], ks_mean[i], 'o', color=colors[i], markersize=11,
                markeredgecolor='white', markeredgewidth=0.9, zorder=3)
        # Place number label offset above-right
        ax.annotate(labels[i],
                    xy=(auroc_mean[i], ks_mean[i]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold', color=colors[i],
                    fontfamily='sans-serif',
                    bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=colors[i],
                              lw=0.9, alpha=0.88),
                    zorder=5)

    ax.set_xlabel("Mean AUROC", fontsize=14)
    ax.set_ylabel("Mean KS Statistic", fontsize=14)
    ax.set_title("AUROC vs. KS — Method Comparison with 95% CIs",
                 fontsize=15, fontweight='bold', pad=12)
    ax.tick_params(labelsize=11)

    # --- Legend panel (right side) ----
    ax_leg.axis('off')
    ax_leg.set_title("Legend", fontsize=14, fontweight='bold', pad=10)

    # Build legend rows: number badge, colored dot, method name
    y_start = 0.92
    y_step = 0.075
    for i, m in enumerate(methods):
        y = y_start - i * y_step
        cat = METHOD_TYPE[m]
        c = TYPE_COLORS[cat]
        # Number badge
        ax_leg.text(0.05, y, labels[i], fontsize=12, fontweight='bold',
                    color=c, transform=ax_leg.transAxes, va='center',
                    ha='center', fontfamily='sans-serif',
                    bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=c,
                              lw=0.9, alpha=0.88))
        # Colored dot
        ax_leg.plot(0.17, y, 'o', color=c, markersize=8,
                    transform=ax_leg.transAxes, markeredgecolor='white',
                    markeredgewidth=0.6, clip_on=False)
        # Method name
        ax_leg.text(0.25, y, m, fontsize=11, color='#333333',
                    transform=ax_leg.transAxes, va='center')

    # Category sub-legend at bottom
    y_bot = y_start - len(methods) * y_step - 0.04
    for j, (cat, c) in enumerate(TYPE_COLORS.items()):
        yy = y_bot - j * 0.055
        ax_leg.plot(0.17, yy, 's', color=c, markersize=8,
                    transform=ax_leg.transAxes, clip_on=False)
        ax_leg.text(0.25, yy, cat, fontsize=10.5, color='#555555',
                    transform=ax_leg.transAxes, va='center')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_auroc_vs_ks_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Pairwise CI Overlap Heatmap (AUROC)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_ci_overlap_heatmap():
    n = len(methods)
    overlap = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            lo = max(auroc_lo[i], auroc_lo[j])
            hi = min(auroc_hi[i], auroc_hi[j])
            total = min(auroc_hi[i] - auroc_lo[i], auroc_hi[j] - auroc_lo[j])
            if total > 0 and hi > lo:
                overlap[i, j] = (hi - lo) / total * 100
            elif i == j:
                overlap[i, j] = 100
            else:
                overlap[i, j] = 0

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(overlap, cmap=cmap, vmin=0, vmax=100, aspect='equal')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(methods, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(methods, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = overlap[i, j]
            color = 'white' if val < 30 or val > 85 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='CI Overlap (%)')
    ax.set_title("Pairwise AUROC CI Overlap (%)\nGreen = Not Significantly Different",
                 fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_ci_overlap_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures...\n")
    print("[1/7] AUROC forest plot...")
    fig_auroc_forest()
    print("[2/7] KS forest plot...")
    fig_ks_forest()
    print("[3/7] CI width stability chart...")
    fig_ci_width()
    print("[4/7] AUROC vs. KS scatter...")
    fig_auroc_vs_ks()
    print("[5/7] CI overlap heatmap...")
    fig_ci_overlap_heatmap()

    # ─── Bundle external Legacy Figures ──────────────────────────────────────────
    print("[6/7] Bundling fig7: P-Value statistical significance heatmap...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pval_src1 = os.path.join(base_dir, "Old figures", "bootstrap_statistical_analysis.jpg")
    pval_src2 = os.path.join(base_dir, "bootstrap_statistical_analysis.jpg")
    pval_dst = os.path.join(OUT_DIR, "fig7_bootstrap_statistical_analysis.jpg")
    
    if os.path.exists(pval_src1):
        shutil.copy(pval_src1, pval_dst)
    elif os.path.exists(pval_src2):
        shutil.copy(pval_src2, pval_dst)
    else:
        print(f"      Warning: Source not found for p-value heatmap")

    print(f"\n✅ All figures saved and bundled to: {OUT_DIR}")
