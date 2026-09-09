"""
Sequence-Level Multi-Gate Mixture of Experts (MoE) ASR Model with Subsampling Frontend
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_framework.models import ExpertPool, TaskGate


class Conv1DSubsampling(nn.Module):
    """
    Standard ASR 2x Temporal Convolutional Subsampling block:
      Input:  (B, T_raw, in_dim)
      Output: (B, T_sub, out_dim), where T_sub = floor((T_raw + 1) / 2)
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is (B, T, D) -> transpose to (B, D, T) for Conv1d
        x_trans = x.transpose(1, 2)
        h = self.conv(x_trans).transpose(1, 2)  # (B, T_sub, out_dim)
        h = self.act(h)
        h = self.norm(h)
        h = self.dropout(h)

        # Compute exact post-subsampling valid lengths
        sub_lengths = torch.div(input_lengths + 1, 2, rounding_mode="floor")
        return h, sub_lengths


class SequenceMultiGateMoE(nn.Module):
    """
    Sequence-level Multi-Gate MoE ASR model trained with Connectionist Temporal Classification (CTC):
      1. Acoustic Subsampling Frontend: 2x Conv1d temporal stride.
      2. Subsampled Frame-wise Multi-Gate MoE Layer: Dynamic top-k sparse routing over shared expert pool.
      3. Temporal Aggregator: Bidirectional LSTM context block.
      4. CTC Projection Head: Maps representations to CTC vocabulary (blank=0, 1..V).
    """

    def __init__(
        self,
        in_dim: int,
        n_vocab: int,  # number of non-blank phonemes/characters
        n_routed_experts: int = 5,
        n_shared_experts: int = 0,
        top_k: int = 2,
        hidden_dim: int = 32,
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

        # 1. 2x Temporal Convolutional Subsampling Frontend
        self.subsampling = Conv1DSubsampling(in_dim=in_dim, out_dim=hidden_dim, dropout=dropout)

        # 2. Routed & Shared Expert Pools
        self.routed_pool = ExpertPool(
            n_experts=n_routed_experts,
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            out_dim=hidden_dim,
            activation=activation,
            dropout=dropout,
        )

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

        # 3. Router Gate
        self.gate = TaskGate(
            in_dim=hidden_dim,
            n_experts=n_routed_experts,
            gate_type=gate_type,
            tasks=tasks,
            acoustic_dim=acoustic_dim,
        )

        # 4. Temporal Sequence Aggregator (BiLSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # 5. CTC Projection Head
        self.ctc_head = nn.Linear(hidden_dim, self.total_vocab)

        # Initialize blank bias to prevent early premature blank collapse
        if init_blank_bias is not None:
            with torch.no_grad():
                self.ctc_head.bias[0] = float(init_blank_bias)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        task: Optional[str] = None,
        acoustic_emb: Optional[torch.Tensor] = None,
        dynamic_bias: Optional[torch.Tensor] = None,
        ablate_expert_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Acoustic sequence tensor of shape (B, T_raw, in_dim)
            input_lengths: (B,) tensor with raw sequence lengths
            task: Task string identifier
            acoustic_emb: Utterance-level acoustic vector (B, acoustic_dim)
            dynamic_bias: Optional load bias (E,)
            ablate_expert_ids: List of expert indices to mask out
        Returns:
            Dict containing:
              - logits: (B, T_sub, total_vocab)
              - log_probs_ctc: (T_sub, B, total_vocab) for nn.CTCLoss
              - sub_lengths: (B,) valid post-subsampling lengths
              - Gm: Frame-level dispatch weights (B*T_sub, E)
              - G: Frame-level dense softmax (B*T_sub, E)
              - mask: Frame-level selection mask (B*T_sub, E)
        """
        # 1. 2x Subsampling: (B, T_raw, in_dim) -> (B, T_sub, hidden_dim)
        h, sub_lengths = self.subsampling(x, input_lengths)
        B, T_sub, H = h.shape

        # Flatten subsampled frames for expert routing: (B*T_sub, hidden_dim)
        h_flat = h.reshape(B * T_sub, self.hidden_dim)

        # Broadcast acoustic embedding across time steps if present
        z_flat = None
        if acoustic_emb is not None:
            z_expanded = acoustic_emb.unsqueeze(1).expand(-1, T_sub, -1)
            z_flat = z_expanded.reshape(B * T_sub, -1)

        # Expert routing
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

        # Compute expert outputs
        routed_outs = self.routed_pool(h_flat)  # (B*T_sub, E, hidden_dim)
        M_routed = torch.einsum("be,bed->bd", Gm, routed_outs)  # (B*T_sub, hidden_dim)

        if self.shared_pool is not None:
            shared_outs = self.shared_pool(h_flat)  # (B*T_sub, N_shared, hidden_dim)
            M_total = M_routed + shared_outs.mean(dim=1)
        else:
            M_total = M_routed

        # Reshape back to sequence: (B, T_sub, hidden_dim)
        moe_seq = M_total.reshape(B, T_sub, self.hidden_dim)

        # Temporal aggregator: (B, T_sub, hidden_dim)
        ctx_seq, _ = self.temporal_encoder(moe_seq)

        # CTC Output Logits: (B, T_sub, total_vocab)
        logits = self.ctc_head(ctx_seq)

        # Log probabilities for PyTorch CTCLoss: shape (T_sub, B, total_vocab)
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
