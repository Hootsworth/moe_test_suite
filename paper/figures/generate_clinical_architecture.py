"""
Generates publication-quality architecture diagram for Neuromotor-Decoupled Clinical MoE
with G_control, G_mild, G_moderate, G_severe labels.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_diagram(output_pdf="paper/figures/architecture_diagram.pdf"):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    c_in = "#E0F2FE"       # Light blue
    c_sub = "#BAE6FD"      # Soft sky blue
    c_gate = "#FEF08A"     # Soft yellow
    c_bias = "#FDE68A"     # Amber yellow
    c_shared = "#BBF7D0"   # Light green
    c_routed = "#DDD6FE"   # Light purple
    c_agg = "#FED7AA"      # Light orange
    c_bilstm = "#FBCFE8"   # Light pink
    c_out = "#E2E8F0"      # Light slate

    # 1. Input Spectrogram
    r1 = patches.FancyBboxPatch((0.4, 2.5), 1.8, 2.2, boxstyle="round,pad=0.1", fc=c_in, ec="#0284C7", lw=1.5)
    ax.add_patch(r1)
    ax.text(1.3, 3.8, "Acoustic Input\nUtterance $X$", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(1.3, 3.1, "$T \\times 80$ Log-Mel\nFilterbanks", ha="center", va="center", fontsize=9, color="#0369A1")

    # Arrow 1 -> Subsampling
    ax.annotate("", xy=(2.7, 3.6), xytext=(2.2, 3.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))

    # 2. 2x Subsampling
    r2 = patches.FancyBboxPatch((2.7, 2.7), 1.6, 1.8, boxstyle="round,pad=0.1", fc=c_sub, ec="#0284C7", lw=1.5)
    ax.add_patch(r2)
    ax.text(3.5, 3.8, "2x Conv1D\nSubsample", ha="center", va="center", fontsize=10, weight="bold")
    ax.text(3.5, 3.1, "$T_{\\text{sub}} \\times D_h$\n($D_h=48$)", ha="center", va="center", fontsize=9, color="#0369A1")

    # Split arrows: to Frame Router and Acoustic Average
    # Frame h_t -> Gate
    ax.annotate("", xy=(4.9, 3.6), xytext=(4.3, 3.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
    # Frame h_t -> Temporal Mean
    ax.plot([4.5, 4.5, 4.9], [3.6, 5.6, 5.6], color="#D97706", lw=1.5, ls="--")
    ax.annotate("", xy=(4.9, 5.6), xytext=(4.8, 5.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))

    # 3. Acoustic Average & Speaker Calibration Projector
    r_bias = patches.FancyBboxPatch((4.9, 4.9), 2.2, 1.4, boxstyle="round,pad=0.1", fc=c_bias, ec="#D97706", lw=1.5)
    ax.add_patch(r_bias)
    ax.text(6.0, 5.8, "Speaker Calibration", ha="center", va="center", fontsize=10, weight="bold", color="#B45309")
    ax.text(6.0, 5.3, "Mean: $\\bar{\\mathbf{h}} = \\frac{1}{T}\\sum h_t$\nBias: $\\beta = W_{\\text{bias}} \\bar{\\mathbf{h}}$", ha="center", va="center", fontsize=8.5)

    # 4. Severity-Decoupled Gating
    r3 = patches.FancyBboxPatch((4.9, 2.3), 2.2, 2.2, boxstyle="round,pad=0.1", fc=c_gate, ec="#CA8A04", lw=1.5)
    ax.add_patch(r3)
    ax.text(6.0, 4.1, "Severity-Decoupled\nRouter Gates $G_\\tau$", ha="center", va="center", fontsize=10, weight="bold", color="#854D0E")
    ax.text(6.0, 3.4, "$G_{\\text{control}}, G_{\\text{mild}}$\n$G_{\\text{moderate}}, G_{\\text{severe}}$", ha="center", va="center", fontsize=8.5, color="#713F12")
    ax.text(6.0, 2.7, "Top-$k$ ($k=2$) Sparse", ha="center", va="center", fontsize=8, color="#854D0E", style="italic")

    # Connect Speaker Bias to Router
    ax.annotate("", xy=(6.0, 4.5), xytext=(6.0, 4.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))
    ax.text(6.25, 4.7, "+ $\\beta$", color="#B45309", fontsize=9, weight="bold")

    # Arrows to Experts
    # 1) Direct to Shared Invariant Expert (unconditional)
    ax.plot([4.5, 4.5, 7.8], [3.6, 1.2, 1.2], color="#059669", lw=1.5, ls="-")
    ax.annotate("", xy=(7.8, 1.2), xytext=(7.7, 1.2), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))

    # 2) Router to Routed Experts
    ax.annotate("", xy=(7.8, 3.6), xytext=(7.1, 3.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#6D28D9"))
    ax.text(7.45, 3.8, "$g_{\\tau, i}$", color="#6D28D9", fontsize=10, weight="bold")

    # 5. Shared Invariant Expert Box
    r_shared = patches.FancyBboxPatch((7.8, 0.6), 2.5, 1.2, boxstyle="round,pad=0.1", fc=c_shared, ec="#059669", lw=1.5)
    ax.add_patch(r_shared)
    ax.text(9.05, 1.35, "Canonical-Anchor Shared Expert", ha="center", va="center", fontsize=9.5, weight="bold", color="#065F46")
    ax.text(9.05, 0.95, "$N_s=1$ Invariant Backbone", ha="center", va="center", fontsize=8.5, color="#047857")

    # 6. Neuromotor Compensatory Routed Experts Box
    r_routed = patches.FancyBboxPatch((7.8, 2.2), 2.5, 2.8, boxstyle="round,pad=0.1", fc=c_routed, ec="#7C3AED", lw=1.5)
    ax.add_patch(r_routed)
    ax.text(9.05, 4.6, "Neuromotor Experts Pool", ha="center", va="center", fontsize=10, weight="bold", color="#5B21B6")
    ax.text(9.05, 4.15, "$E_1$: Formant Centralization Exp.", ha="center", va="center", fontsize=8, color="#4C1D95")
    ax.text(9.05, 3.65, "$E_2$: Temporal Sluggishness Exp.", ha="center", va="center", fontsize=8, color="#4C1D95")
    ax.text(9.05, 3.15, "$E_3$: Glottal/Breathiness Exp.", ha="center", va="center", fontsize=8, color="#4C1D95")
    ax.text(9.05, 2.65, "$E_4$: Articulatory Undershoot Exp.", ha="center", va="center", fontsize=8, color="#4C1D95")

    # Arrows to Aggregation
    ax.annotate("", xy=(10.7, 3.4), xytext=(10.3, 3.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))
    ax.annotate("", xy=(10.7, 3.0), xytext=(10.3, 1.2), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))

    # 7. Aggregation & Sequence Encoder
    r_agg = patches.FancyBboxPatch((10.7, 2.2), 1.4, 1.8, boxstyle="round,pad=0.1", fc=c_agg, ec="#EA580C", lw=1.5)
    ax.add_patch(r_agg)
    ax.text(11.4, 3.3, "Additive Fusion", ha="center", va="center", fontsize=9.5, weight="bold", color="#9A3412")
    ax.text(11.4, 2.7, "$M_{\\text{total}} =$\n$M_{\\text{routed}} + E_{\\text{shared}}$", ha="center", va="center", fontsize=8, color="#C2410C")

    # Arrow to BiLSTM & CTC Head
    ax.annotate("", xy=(12.4, 3.1), xytext=(12.1, 3.1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#334155"))

    # 8. BiLSTM Contextualizer + CTC Output
    r_out = patches.FancyBboxPatch((12.4, 1.8), 1.3, 2.6, boxstyle="round,pad=0.1", fc=c_bilstm, ec="#DB2777", lw=1.5)
    ax.add_patch(r_out)
    ax.text(13.05, 3.9, "BiLSTM", ha="center", va="center", fontsize=10, weight="bold", color="#9D174D")
    ax.text(13.05, 3.3, "Temporal\nContext", ha="center", va="center", fontsize=8.5, color="#BE185D")
    ax.plot([12.5, 13.6], [2.8, 2.8], color="#DB2777", lw=1, ls=":")
    ax.text(13.05, 2.4, "CTC Head", ha="center", va="center", fontsize=9.5, weight="bold", color="#9D174D")
    ax.text(13.05, 2.0, "$p(Y|X)$", ha="center", va="center", fontsize=8.5, color="#BE185D")

    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()
    print(f"Generated updated architecture diagram: {output_pdf}")

if __name__ == "__main__":
    draw_diagram()
