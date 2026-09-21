"""
Neuromotor-Decoupled Clinical MoE Architecture with Speaker Acoustic-Average Calibration
========================================================================================
Designed specifically for atypical and dysarthric speech recognition:
  1. Canonical-Anchor Shared Invariant Expert Pool (Universal phonetics).
  2. Neuromotor Compensatory Sparse Routed Expert Pool (Pathological compensation).
  3. Severity-Decoupled Gating (Control, Mild, Moderate, Severe).
  4. Speaker Acoustic-Average Calibration (Zero-shot VTLN & vowel-space offset).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_framework.models import ExpertPool
from moe_framework.sequence_models import Conv1DSubsampling


class AtypicalClinicalMoE(nn.Module):
    """
    Sequence Mixture-of-Experts for Atypical / Pathological Speech Recognition.
    """

    def __init__(
        self,
        in_dim: int = 80,
        n_vocab: int = 27,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        hidden_dim: int = 48,
        routing_mode: str = "calibrated_atypical",
        severities: Optional[List[str]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.routing_mode = routing_mode
        self.severities = severities or ["control_typical", "mild_dysarthria", "moderate_dysarthria", "severe_dysarthria"]
        self.total_vocab = n_vocab + 1  # 0 is CTC blank

        # 1. 2x Subsampling Frontend
        self.subsampling = Conv1DSubsampling(in_dim=in_dim, out_dim=hidden_dim, dropout=dropout)

        # 2. Canonical-Anchor Shared Invariant Expert Pool
        if n_shared_experts > 0:
            self.shared_pool = ExpertPool(
                n_experts=n_shared_experts,
                in_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                out_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            self.shared_pool = None

        # 3. Neuromotor Compensatory Routed Expert Pool
        self.routed_pool = ExpertPool(
            n_experts=n_routed_experts,
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        # 4. Router Architecture
        if routing_mode in ["decoupled_severity", "calibrated_atypical"]:
            self.severity_gates = nn.ModuleDict({
                sev: nn.Linear(hidden_dim, n_routed_experts) for sev in self.severities
            })
            if routing_mode == "calibrated_atypical":
                # Speaker Acoustic-Average Calibration Projectors
                self.speaker_calib_gates = nn.ModuleDict({
                    sev: nn.Linear(hidden_dim, n_routed_experts, bias=False) for sev in self.severities
                })
        elif routing_mode == "matched_single_gate":
            # 2-layer MLP parameter-matched to 4 linear severity gates (4 * 48 * 4 = 768 params)
            hidden_mlp = max(8, (len(self.severities) * hidden_dim * n_routed_experts) // (hidden_dim + n_routed_experts))
            self.single_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_mlp),
                nn.ReLU(),
                nn.Linear(hidden_mlp, n_routed_experts),
            )
        elif routing_mode == "standard_single_gate":
            self.single_gate = nn.Linear(hidden_dim, n_routed_experts)
        else:
            raise ValueError(f"Unknown routing_mode: {routing_mode}")

        # 5. Temporal Sequence Contextualizer (BiLSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 6. CTC Projection Head
        self.ctc_head = nn.Linear(hidden_dim, self.total_vocab)
        with torch.no_grad():
            self.ctc_head.bias[0] = -2.0  # Encourage initial token predictions

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        shared = sum(p.numel() for p in self.shared_pool.parameters() if p.requires_grad) if self.shared_pool else 0
        routed = sum(p.numel() for p in self.routed_pool.parameters() if p.requires_grad)
        return {
            "total": total,
            "shared_anchor": shared,
            "routed_compensators": routed,
        }

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        severity: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        h, sub_lengths = self.subsampling(x, input_lengths)
        B, T_sub, H = h.shape

        h_flat = h.reshape(B * T_sub, H)

        # Route Computation
        if self.routing_mode in ["decoupled_severity", "calibrated_atypical"]:
            sev = severity or self.severities[0]
            raw_logits = self.severity_gates[sev](h_flat)

            if self.routing_mode == "calibrated_atypical":
                # Compute utterance acoustic average per batch item: (B, H)
                mask_2d = torch.arange(T_sub, device=h.device).unsqueeze(0) < sub_lengths.unsqueeze(1)
                mask_3d = mask_2d.unsqueeze(-1).float()
                h_sum = (h * mask_3d).sum(dim=1)
                denom = sub_lengths.unsqueeze(-1).float().clamp(min=1.0)
                h_bar = h_sum / denom  # (B, H)

                # Project to speaker calibration bias: (B, N_experts)
                beta_patient = self.speaker_calib_gates[sev](h_bar)
                beta_expanded = beta_patient.unsqueeze(1).expand(-1, T_sub, -1).reshape(B * T_sub, -1)
                logits = raw_logits + beta_expanded
            else:
                logits = raw_logits

        elif self.routing_mode in ["standard_single_gate", "matched_single_gate"]:
            logits = self.single_gate(h_flat)

        # Top-k sparse dispatch
        G = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(G, k=self.top_k, dim=-1)
        denom = topk_vals.sum(dim=-1, keepdim=True) + 1e-9
        norm_topk_vals = topk_vals / denom

        Gm = torch.zeros_like(G)
        Gm.scatter_(1, topk_idx, norm_topk_vals)

        mask = torch.zeros_like(G)
        mask.scatter_(1, topk_idx, 1.0)

        # Execute Neuromotor Compensatory Routed Experts
        routed_outs = self.routed_pool(h_flat)  # (B*T_sub, N_routed, H)
        M_routed = torch.einsum("be,bed->bd", Gm, routed_outs)

        # Execute Canonical-Anchor Shared Invariant Experts
        if self.shared_pool is not None:
            shared_outs = self.shared_pool(h_flat)  # (B*T_sub, N_shared, H)
            M_shared = shared_outs.mean(dim=1)
            M_total = M_routed + M_shared
        else:
            M_total = M_routed

        moe_seq = M_total.reshape(B, T_sub, H)
        ctx_seq, _ = self.temporal_encoder(moe_seq)
        logits_ctc = self.ctc_head(ctx_seq)
        log_probs_ctc = F.log_softmax(logits_ctc, dim=-1).transpose(0, 1)

        return {
            "logits": logits_ctc,
            "log_probs_ctc": log_probs_ctc,
            "sub_lengths": sub_lengths,
            "Gm": Gm,
            "G": G,
            "mask": mask,
            "indices": topk_idx,
        }
