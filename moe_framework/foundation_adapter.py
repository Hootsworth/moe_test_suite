"""
Self-Supervised Foundation Model Clinical MoE Adapter Architecture
===================================================================
Integrates the Neuromotor-Decoupled Clinical MoE with Speaker
Acoustic-Average Calibration on top of frozen Self-Supervised Speech
Foundation Encoders (Wav2Vec 2.0 / HuBERT / WavLM).

Key Properties:
  1. Frozen SSL Backbone: Preserves pre-trained phonetic representations (94.3M params frozen).
  2. Parameter-Efficient Adapter: Only adapter and routing parameters are trained (~135k params, 0.14%).
  3. Decoupled Severity Router: Separate routing policies per clinical stratum.
  4. Speaker Acoustic-Average Calibration: Normalizes high-level SSL embedding drift via beta = W_bias * h_bar.
  5. Canonical Anchor + Compensatory Expert Pools: Isolates invariant phonetics from pathological compensation.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from moe_framework.models import ExpertPool


class FoundationClinicalMoEAdapter(nn.Module):
    """
    Parameter-Efficient Clinical Mixture-of-Experts Adapter for Speech Foundation Models.
    """

    def __init__(
        self,
        backbone_name: str = "WAV2VEC2_BASE",
        adapter_dim: int = 64,
        n_vocab: int = 27,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        routing_mode: str = "calibrated_atypical",
        severities: Optional[List[str]] = None,
        freeze_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.adapter_dim = adapter_dim
        self.n_vocab = n_vocab
        self.total_vocab = n_vocab + 1  # 0 is CTC blank
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.routing_mode = routing_mode
        self.severities = severities or ["control_typical", "mild_dysarthria", "moderate_dysarthria", "severe_dysarthria"]
        self.freeze_backbone = freeze_backbone

        # 1. Load Pretrained Foundation Backbone (Wav2Vec 2.0 Base: 768-dim features)
        bundle = getattr(torchaudio.pipelines, backbone_name, torchaudio.pipelines.WAV2VEC2_BASE)
        self.backbone = bundle.get_model()
        self.feature_dim = 768

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # 2. Adapter Down-Projection
        self.down_proj = nn.Sequential(
            nn.Linear(self.feature_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Routing & Expert Pools (skipped if linear_probe)
        if routing_mode == "linear_probe":
            self.shared_pool = None
            self.routed_pool = None
            self.single_gate = None
            self.severity_gates = None
            self.speaker_calib_gates = None
        else:
            # Canonical Anchor Shared Invariant Expert Pool
            if n_shared_experts > 0 and routing_mode != "standard_single_gate":
                self.shared_pool = ExpertPool(
                    n_experts=n_shared_experts,
                    in_dim=adapter_dim,
                    hidden_dim=adapter_dim * 2,
                    out_dim=adapter_dim,
                    dropout=dropout,
                )
            else:
                self.shared_pool = None

            # Neuromotor Compensatory Routed Expert Pool
            self.routed_pool = ExpertPool(
                n_experts=n_routed_experts,
                in_dim=adapter_dim,
                hidden_dim=adapter_dim * 2,
                out_dim=adapter_dim,
                dropout=dropout,
            )

            # Routing Mechanisms
            if routing_mode in ["decoupled_severity", "calibrated_atypical"]:
                self.severity_gates = nn.ModuleDict({
                    sev: nn.Linear(adapter_dim, n_routed_experts) for sev in self.severities
                })
                if routing_mode == "calibrated_atypical":
                    self.speaker_calib_gates = nn.ModuleDict({
                        sev: nn.Linear(adapter_dim, n_routed_experts, bias=False) for sev in self.severities
                    })
                else:
                    self.speaker_calib_gates = None
                self.single_gate = None

            elif routing_mode == "standard_single_gate":
                self.single_gate = nn.Linear(adapter_dim, n_routed_experts)
                self.severity_gates = None
                self.speaker_calib_gates = None
            else:
                raise ValueError(f"Unknown routing_mode: {routing_mode}")

        # 4. Up-Projection and CTC Head
        self.up_proj = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ctc_head = nn.Linear(adapter_dim, self.total_vocab)
        with torch.no_grad():
            self.ctc_head.bias[0] = -2.0  # CTC blank suppression initialization

    def count_parameters(self) -> Dict[str, int]:
        total_backbone = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "total_backbone": total_backbone,
            "trainable_adapter": trainable,
            "frozen": frozen,
            "trainable_pct": (trainable / (trainable + frozen)) * 100.0 if (trainable + frozen) > 0 else 0.0,
        }

    def extract_backbone_features(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes raw audio waveforms (16kHz) through frozen Wav2Vec2 backbone.
        waveforms: (B, L)
        lengths: (B,)
        Returns: contextual representations (B, T, 768), subsampled lengths (B,)
        """
        if self.freeze_backbone:
            with torch.no_grad():
                features, out_lengths = self.backbone(waveforms, lengths)
        else:
            features, out_lengths = self.backbone(waveforms, lengths)
        return features, out_lengths

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        severity: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Clinical MoE Adapter.
        features: (B, T, 768) - can be pre-extracted or passed from backbone
        lengths: (B,)
        """
        B, T, D = features.shape

        # 1. Adapter Down-Projection: (B, T, adapter_dim)
        h = self.down_proj(features)
        h_flat = h.reshape(B * T, self.adapter_dim)

        # 2. Linear Probe Baseline
        if self.routing_mode == "linear_probe":
            moe_out = h
            G = torch.ones(B * T, 1, device=features.device)
            mask = torch.ones_like(G)
        else:
            # 3. Routing Computation
            if self.routing_mode in ["decoupled_severity", "calibrated_atypical"]:
                sev = severity or self.severities[0]
                raw_logits = self.severity_gates[sev](h_flat)

                if self.routing_mode == "calibrated_atypical":
                    # Utterance acoustic-average vector in adapter embedding space: (B, adapter_dim)
                    mask_2d = torch.arange(T, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
                    mask_3d = mask_2d.unsqueeze(-1).float()
                    h_sum = (h * mask_3d).sum(dim=1)
                    denom = lengths.unsqueeze(-1).float().clamp(min=1.0)
                    h_bar = h_sum / denom  # (B, adapter_dim)

                    # Project speaker bias vector: (B, N_routed)
                    beta_spk = self.speaker_calib_gates[sev](h_bar)
                    beta_expanded = beta_spk.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                    logits = raw_logits + beta_expanded
                else:
                    logits = raw_logits

            elif self.routing_mode == "standard_single_gate":
                logits = self.single_gate(h_flat)

            # Top-k sparse routing dispatch
            G = F.softmax(logits, dim=-1)
            topk_vals, topk_idx = torch.topk(G, k=min(self.top_k, self.n_routed_experts), dim=-1)
            norm_topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

            Gm = torch.zeros_like(G)
            Gm.scatter_(1, topk_idx, norm_topk_vals)

            mask = torch.zeros_like(G)
            mask.scatter_(1, topk_idx, 1.0)

            # Routed Experts Execution
            routed_outs = self.routed_pool(h_flat)  # (B*T, N_routed, adapter_dim)
            M_routed = torch.einsum("be,bed->bd", Gm, routed_outs)

            # Canonical Anchor Shared Expert Execution
            if self.shared_pool is not None:
                shared_outs = self.shared_pool(h_flat)  # (B*T, N_shared, adapter_dim)
                M_shared = shared_outs.mean(dim=1)
                M_total = M_routed + M_shared
            else:
                M_total = M_routed

            moe_out = M_total.reshape(B, T, self.adapter_dim)

        # 4. Up-Projection and CTC Head
        proj_out = self.up_proj(moe_out)
        logits_ctc = self.ctc_head(proj_out)  # (B, T, total_vocab)
        log_probs_ctc = F.log_softmax(logits_ctc, dim=-1).transpose(0, 1)  # (T, B, total_vocab)

        return {
            "logits": logits_ctc,
            "log_probs_ctc": log_probs_ctc,
            "out_lengths": lengths,
            "G": G,
            "mask": mask,
        }
