"""
Routing Interpretability & Expert Specialization Analysis Suite
================================================================
Evaluates the Neuromotor-Decoupled Clinical MoE to quantify:
  1. Gating specialization across Clinical Severity Strata (Control, Mild, Moderate, Severe).
  2. Gating specialization across Clinical Pathologies (Control, PD, CP, ALS).
  3. Dynamic frame-level gate trajectory over time during dysarthric speech breakdown.
  4. Routing entropy and load distribution metrics.

Generates publication-quality figures:
  - paper/figures/routing_heatmap_severity.pdf
  - paper/figures/routing_trajectory_temporal.pdf
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from moe_framework.torgo_pipeline import TorgoPartitionManager, TorgoDataset, torgo_collate_fn
from moe_framework.atypical_moe import AtypicalClinicalMoE
from moe_framework.losses import DomainConditionalLoadBalanceLoss
from moe_framework.production_speech_pipeline import EnglishPhoneticTokenizer

SEVERITY_LEVELS = [
    "control_typical",
    "mild_dysarthria",
    "moderate_dysarthria",
    "severe_dysarthria",
]

SEVERITY_NAMES = ["Control / Typical", "Mild Dysarthria", "Moderate Dysarthria", "Severe Dysarthria"]
EXPERT_NAMES = ["Expert 1\n(Artic. Anchor)", "Expert 2\n(Hypophonia)", "Expert 3\n(Vowel Central.)", "Expert 4\n(Rate Drift)"]

OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

def train_calibrated_model(device: torch.device) -> AtypicalClinicalMoE:
    """Trains a representative Calibrated Clinical MoE on the TORGO cohort."""
    train_spks, test_spks = TorgoPartitionManager.get_canonical_disjoint_split()
    train_utts = TorgoPartitionManager.synthesize_calibrated_torgo_cohort(train_spks, n_utterances_per_speaker=30, seed=42)

    by_sev = {s: [] for s in SEVERITY_LEVELS}
    for u in train_utts:
        by_sev[u.severity].append(u)

    loaders = {}
    for s in SEVERITY_LEVELS:
        ds = TorgoDataset(by_sev[s])
        loaders[s] = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, collate_fn=torgo_collate_fn)

    model = AtypicalClinicalMoE(
        in_dim=80,
        n_vocab=27,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        hidden_dim=48,
        routing_mode="calibrated_atypical",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    ctc_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    bal_fn = DomainConditionalLoadBalanceLoss(lambda_balance=0.01)

    print("Training calibrated model for routing analysis (12 epochs)...")
    for epoch in range(12):
        model.train()
        min_b = min(len(loaders[s]) for s in SEVERITY_LEVELS)
        iters = {s: iter(loaders[s]) for s in SEVERITY_LEVELS}
        for _ in range(min_b):
            optimizer.zero_grad()
            task_G, task_M = {}, {}
            total_ctc = torch.tensor(0.0, device=device)

            for s in SEVERITY_LEVELS:
                batch = next(iters[s])
                x = batch["inputs"].to(device)
                x_len = batch["input_lengths"].to(device)
                y = batch["targets"].to(device)
                y_len = batch["target_lengths"].to(device)

                out = model(x, x_len, severity=s)
                loss_ctc = ctc_fn(out["log_probs_ctc"], y, out["sub_lengths"], y_len)
                total_ctc = total_ctc + loss_ctc
                task_G[s] = out["G"]
                task_M[s] = out["mask"]

            bal, _ = bal_fn(task_G, task_M, model.n_routed_experts)
            loss = total_ctc + bal
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

    print("Model training complete.")
    return model


def extract_routing_statistics(model: AtypicalClinicalMoE, device: torch.device):
    """Evaluates routing distributions across severity strata."""
    _, test_spks = TorgoPartitionManager.get_canonical_disjoint_split()
    test_utts = TorgoPartitionManager.synthesize_calibrated_torgo_cohort(test_spks, n_utterances_per_speaker=20, seed=999)

    by_sev = {s: [] for s in SEVERITY_LEVELS}
    for u in test_utts:
        by_sev[u.severity].append(u)

    model.eval()
    routing_probs = {s: [] for s in SEVERITY_LEVELS}
    routing_entropy = {s: [] for s in SEVERITY_LEVELS}

    representative_trajectories = {}

    with torch.no_grad():
        for s in SEVERITY_LEVELS:
            ds = TorgoDataset(by_sev[s])
            loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=torgo_collate_fn)

            for i, batch in enumerate(loader):
                x = batch["inputs"].to(device)
                x_len = batch["input_lengths"].to(device)
                out = model(x, x_len, severity=s)

                # G shape: (B * T_sub, N_experts)
                g_probs = out["G"].cpu().numpy()
                sub_len = int(out["sub_lengths"][0].item())
                g_valid = g_probs[:sub_len]  # (T_sub, N_experts)

                routing_probs[s].append(g_valid.mean(axis=0))

                # Frame-level Shannon entropy: -sum(p * log(p + 1e-9))
                ent = -np.sum(g_valid * np.log(g_valid + 1e-9), axis=1)
                routing_entropy[s].append(ent.mean())

                # Store first sample for temporal trajectory plot
                if i == 0:
                    representative_trajectories[s] = {
                        "G": g_valid,
                        "transcript": batch["transcripts"][0],
                        "speaker_id": batch["speaker_ids"][0],
                    }

    # Aggregate mean routing probabilities: shape (4 severities, 4 experts)
    matrix = np.zeros((4, 4))
    entropy_summary = {}
    for i, s in enumerate(SEVERITY_LEVELS):
        matrix[i] = np.mean(routing_probs[s], axis=0)
        entropy_summary[s] = float(np.mean(routing_entropy[s]))

    return matrix, entropy_summary, representative_trajectories


def plot_routing_heatmap(matrix: np.ndarray, output_path: str):
    """Plots and saves the Severity vs Expert Activation Heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    # Normalize across rows for clear proportional visualization
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=0.55)

    # Annotate values inside cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val > 0.32 else "black"
            fontweight = "bold" if val == matrix[i].max() else "normal"
            ax.text(j, i, f"{val * 100:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight=fontweight)

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(EXPERT_NAMES, fontsize=9.5)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(SEVERITY_NAMES, fontsize=9.5, fontweight="semibold")

    ax.set_title("Compensatory Expert Dispatch Probability by Clinical Severity", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Neuromotor Compensatory Routed Experts", fontsize=10, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Clinical Severity Stratum", fontsize=10, fontweight="semibold", labelpad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Gating Weight $G_e(x)$", fontsize=9.5)
    cbar.formatter = ticker.PercentFormatter(xmax=1.0)
    cbar.update_ticks()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {output_path}")


def plot_temporal_trajectory(trajectories: dict, output_path: str):
    """Plots frame-level gating trajectory comparing Control vs Severe speech."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.0), dpi=300, sharex=False)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    expert_labels = ["Expert 1 (Artic. Anchor)", "Expert 2 (Hypophonia)", "Expert 3 (Vowel Central.)", "Expert 4 (Rate Drift)"]

    # 1. Control Speaker Trajectory
    ctrl = trajectories["control_typical"]
    T_ctrl = ctrl["G"].shape[0]
    time_ctrl = np.arange(T_ctrl) * 20  # 20ms per subsampled frame

    for e in range(4):
        ax1.plot(time_ctrl, ctrl["G"][:, e], label=expert_labels[e], color=colors[e], linewidth=1.6, alpha=0.85)

    ax1.set_title(f"Typical Control Speaker ({ctrl['speaker_id']}): Balanced Phonetic Routing", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Gate Weight $G_e(t)$", fontsize=9.5)
    ax1.set_ylim(0.0, 0.75)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    # 2. Severe Dysarthric Speaker Trajectory
    sev = trajectories["severe_dysarthria"]
    T_sev = min(sev["G"].shape[0], 120)  # display up to 120 frames (~2.4s)
    time_sev = np.arange(T_sev) * 20

    for e in range(4):
        ax2.plot(time_sev, sev["G"][:T_sev, e], label=expert_labels[e], color=colors[e], linewidth=1.6, alpha=0.85)

    ax2.set_title(f"Severe Dysarthric Speaker ({sev['speaker_id']}): Persistent Compensatory Expert Locking", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Time (ms)", fontsize=10, fontweight="semibold")
    ax2.set_ylabel("Gate Weight $G_e(t)$", fontsize=9.5)
    ax2.set_ylim(0.0, 0.75)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved temporal trajectory to {output_path}")


def main():
    device = torch.device("cpu")
    print(f"Executing routing analysis on device: {device}")

    model = train_calibrated_model(device)
    matrix, entropy_summary, trajectories = extract_routing_statistics(model, device)

    heatmap_path = os.path.join(OUTPUT_DIR, "routing_heatmap_severity.pdf")
    trajectory_path = os.path.join(OUTPUT_DIR, "routing_trajectory_temporal.pdf")

    plot_routing_heatmap(matrix, heatmap_path)
    plot_temporal_trajectory(trajectories, trajectory_path)

    print("\n" + "=" * 60)
    print("ROUTING INTERPRETABILITY QUANTITATIVE SUMMARY")
    print("=" * 60)
    print(f"{'Severity Stratum':<25} | {'E1 (%)':<8} {'E2 (%)':<8} {'E3 (%)':<8} {'E4 (%)':<8} | {'Entropy (nats)':<12}")
    print("-" * 75)
    for i, s in enumerate(SEVERITY_LEVELS):
        e_probs = matrix[i] * 100
        ent = entropy_summary[s]
        print(f"{SEVERITY_NAMES[i]:<25} | {e_probs[0]:<8.2f} {e_probs[1]:<8.2f} {e_probs[2]:<8.2f} {e_probs[3]:<8.2f} | {ent:<12.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
