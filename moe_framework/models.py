import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feedforward activation block."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_val = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        val = self.w_val(x)
        return self.w_out(gate * val)


class Expert(nn.Module):
    """Feedforward Expert module with customizable activations (ReLU, GELU, SwiGLU)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.activation_type = activation.lower()

        if self.activation_type == "swiglu":
            self.net = SwiGLU(in_dim, hidden_dim, out_dim)
        else:
            act_fn = nn.ReLU() if self.activation_type == "relu" else nn.GELU()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                act_fn,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertPool(nn.Module):
    """Pool of N individual Expert sub-networks."""

    def __init__(
        self,
        n_experts: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            Expert(in_dim, hidden_dim, out_dim, activation=activation, dropout=dropout)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes outputs from all experts.
        Args:
            x: Tensor of shape (B, D)
        Returns:
            Tensor of shape (B, E, D_out)
        """
        outs = [expert(x) for expert in self.experts]
        return torch.stack(outs, dim=1)


class TaskGate(nn.Module):
    """
    Routing Gate supporting:
      1. 'multi_gate': Independent linear router per discrete task ID.
      2. 'single_gate': Shared linear router across all inputs.
      3. 'continuous_acoustic': Router conditioned on both input x and continuous acoustic vector z.
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int,
        gate_type: str = "multi_gate",
        tasks: Optional[List[str]] = None,
        acoustic_dim: int = 0,
        renormalize_topk: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_experts = n_experts
        self.gate_type = gate_type
        self.renormalize_topk = renormalize_topk
        self.acoustic_dim = acoustic_dim

        if self.gate_type == "multi_gate":
            if not tasks:
                raise ValueError("`tasks` list must be provided when gate_type='multi_gate'.")
            self.task_gates = nn.ModuleDict({
                t: nn.Linear(in_dim, n_experts) for t in tasks
            })
        elif self.gate_type == "single_gate":
            self.shared_gate = nn.Linear(in_dim, n_experts)
        elif self.gate_type == "continuous_acoustic":
            total_dim = in_dim + acoustic_dim
            self.acoustic_gate = nn.Sequential(
                nn.Linear(total_dim, max(in_dim // 2, n_experts * 2)),
                nn.ReLU(),
                nn.Linear(max(in_dim // 2, n_experts * 2), n_experts),
            )
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None,
        acoustic_emb: Optional[torch.Tensor] = None,
        dynamic_bias: Optional[torch.Tensor] = None,
        top_k: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for router.
        Returns:
            Gm: (B, E) sparse dispatch weights actually applied.
            G: (B, E) full dense softmax probabilities (for load balancing & metrics).
            indices: (B, k) chosen expert indices.
            mask: (B, E) binary selection mask.
        """
        B = x.shape[0]

        # Compute raw gating logits
        if self.gate_type == "multi_gate":
            if task not in self.task_gates:
                raise ValueError(f"Task '{task}' not found in task gates: {list(self.task_gates.keys())}")
            logits = self.task_gates[task](x)
        elif self.gate_type == "single_gate":
            logits = self.shared_gate(x)
        elif self.gate_type == "continuous_acoustic":
            if acoustic_emb is None:
                raise ValueError("acoustic_emb must be provided for continuous_acoustic gating")
            if acoustic_emb.ndim == 1:
                acoustic_emb = acoustic_emb.unsqueeze(0).expand(B, -1)
            inp = torch.cat([x, acoustic_emb], dim=-1)
            logits = self.acoustic_gate(inp)

        # Optional dynamic bias for auxiliary-loss-free routing (DeepSeek-V3 style)
        if dynamic_bias is not None:
            logits = logits + dynamic_bias.unsqueeze(0)

        # Dense softmax probabilities
        G = F.softmax(logits, dim=-1)

        # Top-k sparse selection
        if top_k >= self.n_experts:
            mask = torch.ones_like(G)
            Gm = G
            indices = torch.arange(self.n_experts, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            topk_vals, topk_indices = torch.topk(G, k=top_k, dim=-1)
            mask = torch.zeros_like(G).scatter_(-1, topk_indices, 1.0)
            if self.renormalize_topk:
                # Renormalize top-k probabilities to sum to 1
                Gm_sparse = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-12)
                Gm = torch.zeros_like(G).scatter_(-1, topk_indices, Gm_sparse)
            else:
                # Switch/GShard style: keep raw softmax probability
                Gm = G * mask
            indices = topk_indices

        return Gm, G, indices, mask


class MultiGateMoEClassifier(nn.Module):
    """
    Complete Multi-Gate MoE classification model supporting:
      - Shared Invariant Experts (DeepSeek hybrid isolation)
      - Routed Sparse Experts (Top-k sparse dispatch)
      - Multi-Gate / Single-Gate / Continuous-Acoustic routing
      - Expert Ablation Masking at inference time
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        n_routed_experts: int = 5,
        n_shared_experts: int = 0,
        top_k: int = 2,
        hidden_dim: int = 24,
        gate_type: str = "multi_gate",
        tasks: Optional[List[str]] = None,
        acoustic_dim: int = 0,
        activation: str = "relu",
        dropout: float = 0.0,
        renormalize_topk: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.gate_type = gate_type

        # Routed expert pool
        self.routed_pool = ExpertPool(
            n_experts=n_routed_experts,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=in_dim,
            activation=activation,
            dropout=dropout,
        )

        # Optional DeepSeek-style shared experts (domain-invariant backbone)
        if n_shared_experts > 0:
            self.shared_pool = ExpertPool(
                n_experts=n_shared_experts,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=in_dim,
                activation=activation,
                dropout=dropout,
            )
        else:
            self.shared_pool = None

        # Routing Gate
        self.gate = TaskGate(
            in_dim=in_dim,
            n_experts=n_routed_experts,
            gate_type=gate_type,
            tasks=tasks,
            acoustic_dim=acoustic_dim,
            renormalize_topk=renormalize_topk,
        )

        # Classification Head
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        task: Optional[str] = None,
        acoustic_emb: Optional[torch.Tensor] = None,
        dynamic_bias: Optional[torch.Tensor] = None,
        ablate_expert_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor (B, D)
            task: Task string (for multi_gate mode)
            acoustic_emb: Optional acoustic embedding tensor (B, D_acoustic)
            dynamic_bias: Optional routing bias (E,)
            ablate_expert_ids: List of expert indices to zero-out (for ablation studies)
        Returns:
            Dict containing logits, probabilities, Gm, G, mask, indices, and expert outputs.
        """
        B = x.shape[0]

        # Routed expert forward: (B, E, D)
        routed_outs = self.routed_pool(x)

        # Router forward
        Gm, G, indices, mask = self.gate(
            x=x,
            task=task,
            acoustic_emb=acoustic_emb,
            dynamic_bias=dynamic_bias,
            top_k=self.top_k,
        )

        # Apply ablation mask if specified
        if ablate_expert_ids:
            for eid in ablate_expert_ids:
                if 0 <= eid < self.n_routed_experts:
                    Gm = Gm.clone()
                    Gm[:, eid] = 0.0

        # Weighted combination of routed experts: (B, D)
        # einsum: Gm(B, E), routed_outs(B, E, D) -> M(B, D)
        M_routed = torch.einsum("be,bed->bd", Gm, routed_outs)

        # Shared experts contribution (if present)
        if self.shared_pool is not None:
            shared_outs = self.shared_pool(x)  # (B, N_shared, D)
            M_shared = shared_outs.mean(dim=1)  # (B, D)
            M_total = M_routed + M_shared
        else:
            M_total = M_routed

        # Classification logits
        logits = self.classifier(M_total)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "Gm": Gm,
            "G": G,
            "mask": mask,
            "indices": indices,
            "routed_outs": routed_outs,
            "M": M_total,
        }
