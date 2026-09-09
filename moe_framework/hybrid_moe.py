"""
DeepSeek-Style Hybrid Mixture of Experts (MoE) ASR Model for Speech Production
with Parameter-Matched Baselines and Architecture Ablations
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_framework.models import ExpertPool, TaskGate
from moe_framework.sequence_models import Conv1DSubsampling


class MatchedSingleGate(nn.Module):
    """
    Parameter-matched single gate baseline:
    Uses a 2-layer MLP to match the total parameter count of 3 separate linear task gates.
    """
    def __init__(self, in_dim: int, n_experts: int, n_tasks: int = 3):
        super().__init__()
        # Total parameters in n_tasks linear gates = n_tasks * in_dim * n_experts
        # MLP: in_dim -> hidden -> n_experts
        # in_dim * hidden + hidden * n_experts = n_tasks * in_dim * n_experts
        hidden = max(8, (n_tasks * in_dim * n_experts) // (in_dim + n_experts))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )
        self.n_experts = n_experts

    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None,
        acoustic_emb: Optional[torch.Tensor] = None,
        dynamic_bias: Optional[torch.Tensor] = None,
        top_k: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        if dynamic_bias is not None:
            logits = logits + dynamic_bias.unsqueeze(0)

        G = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(G, k=top_k, dim=-1)
        denom = topk_vals.sum(dim=-1, keepdim=True) + 1e-9
        norm_topk_vals = topk_vals / denom

        Gm = torch.zeros_like(G)
        Gm.scatter_(1, topk_idx, norm_topk_vals)

        mask = torch.zeros_like(G)
        mask.scatter_(1, topk_idx, 1.0)

        return Gm, G, topk_idx, mask


class HybridSequenceMoE(nn.Module):
    """
    Hybrid Mixture-of-Experts (MoE) Architecture:
      - 2x Temporal Convolutional Subsampling Frontend.
      - N_shared Dedicated Shared Invariant Experts (capturing universal phonetic backbone).
      - N_routed Sparse Routed Experts (domain-specialized vocal-tract/temporal compensators).
      - Bidirectional LSTM temporal context layer.
      - Linear CTC Projection Head.
    """

    def __init__(
        self,
        in_dim: int = 80,
        n_vocab: int = 27,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        hidden_dim: int = 48,
        gate_type: str = "multi_gate",
        tasks: Optional[List[str]] = None,
        acoustic_dim: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
        init_blank_bias: float = -2.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_vocab = n_vocab
        self.total_vocab = n_vocab + 1  # 0 is CTC blank
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.gate_type = gate_type

        # 1. 2x Subsampling Frontend
        self.subsampling = Conv1DSubsampling(in_dim=in_dim, out_dim=hidden_dim, dropout=dropout)

        # 2. Dedicated Shared Invariant Expert Pool
        if n_shared_experts > 0:
            self.shared_pool = ExpertPool(
                n_experts=n_shared_experts,
                in_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                out_dim=hidden_dim,
                activation=activation,
                dropout=dropout,
            )
        else:
            self.shared_pool = None

        # 3. Sparse Routed Expert Pool
        self.routed_pool = ExpertPool(
            n_experts=n_routed_experts,
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )

        # 4. Router Gate
        if gate_type == "matched_single_gate":
            self.gate = MatchedSingleGate(
                in_dim=hidden_dim,
                n_experts=n_routed_experts,
                n_tasks=len(tasks) if tasks else 3,
            )
        else:
            self.gate = TaskGate(
                in_dim=hidden_dim,
                n_experts=n_routed_experts,
                gate_type=gate_type,
                tasks=tasks,
                acoustic_dim=acoustic_dim,
            )

        # 5. Temporal Sequence Aggregator (BiLSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 6. CTC Projection Head
        self.ctc_head = nn.Linear(hidden_dim, self.total_vocab)

        if init_blank_bias is not None:
            with torch.no_grad():
                self.ctc_head.bias[0] = float(init_blank_bias)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frontend = sum(p.numel() for p in self.subsampling.parameters() if p.requires_grad)
        shared = sum(p.numel() for p in self.shared_pool.parameters() if p.requires_grad) if self.shared_pool else 0
        routed = sum(p.numel() for p in self.routed_pool.parameters() if p.requires_grad)
        gate = sum(p.numel() for p in self.gate.parameters() if p.requires_grad)
        temporal = sum(p.numel() for p in self.temporal_encoder.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.ctc_head.parameters() if p.requires_grad)
        return {
            "total": total,
            "frontend": frontend,
            "shared_experts": shared,
            "routed_experts": routed,
            "gate": gate,
            "temporal_encoder": temporal,
            "ctc_head": head,
        }

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        task: Optional[str] = None,
        acoustic_emb: Optional[torch.Tensor] = None,
        dynamic_bias: Optional[torch.Tensor] = None,
        ablate_expert_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        h, sub_lengths = self.subsampling(x, input_lengths)
        B, T_sub, H = h.shape

        h_flat = h.reshape(B * T_sub, self.hidden_dim)

        z_flat = None
        if acoustic_emb is not None:
            z_expanded = acoustic_emb.unsqueeze(1).expand(-1, T_sub, -1)
            z_flat = z_expanded.reshape(B * T_sub, -1)

        # Router forward
        Gm, G, indices, mask = self.gate(
            x=h_flat,
            task=task,
            acoustic_emb=z_flat,
            dynamic_bias=dynamic_bias,
            top_k=self.top_k,
        )

        # Apply ablation mask if specified
        if ablate_expert_ids:
            for eid in ablate_expert_ids:
                if 0 <= eid < self.n_routed_experts:
                    Gm = Gm.clone()
                    Gm[:, eid] = 0.0

        # Routed expert outputs
        routed_outs = self.routed_pool(h_flat)  # (B*T_sub, E_routed, H)
        M_routed = torch.einsum("be,bed->bd", Gm, routed_outs)

        # Shared invariant expert outputs
        if self.shared_pool is not None:
            shared_outs = self.shared_pool(h_flat)  # (B*T_sub, N_shared, H)
            M_shared = shared_outs.mean(dim=1)
            M_total = M_routed + M_shared
        else:
            M_total = M_routed

        moe_seq = M_total.reshape(B, T_sub, self.hidden_dim)
        ctx_seq, _ = self.temporal_encoder(moe_seq)
        logits = self.ctc_head(ctx_seq)
        log_probs_ctc = F.log_softmax(logits, dim=-1).transpose(0, 1)

        return {
            "logits": logits,
            "log_probs_ctc": log_probs_ctc,
            "sub_lengths": sub_lengths,
            "Gm": Gm,
            "G": G,
            "mask": mask,
            "indices": indices,
        }
