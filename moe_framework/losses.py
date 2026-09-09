import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class StandardLoadBalanceLoss(nn.Module):
    """
    Standard Switch / GShard auxiliary load balancing loss:
    L_balance = lambda * E * sum_e (f_e * P_e)
    where f_e is the fraction of tokens routed to expert e (stop-gradient),
    and P_e is the mean softmax routing probability assigned to expert e.
    """

    def __init__(self, lambda_balance: float = 0.02):
        super().__init__()
        self.lambda_balance = lambda_balance

    def forward(
        self,
        G_list: List[torch.Tensor],
        mask_list: List[torch.Tensor],
        n_experts: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            G_list: List of dense gate softmax probability tensors (B_i, E)
            mask_list: List of binary selection masks (B_i, E)
            n_experts: Total number of routed experts E
        Returns:
            loss: Scalar PyTorch loss tensor
            loss_val: Float value for logging
        """
        if not G_list or self.lambda_balance <= 0.0:
            return torch.tensor(0.0, device=G_list[0].device if G_list else "cpu"), 0.0

        all_G = torch.cat(G_list, dim=0)       # (N_total, E)
        all_mask = torch.cat(mask_list, dim=0) # (N_total, E)

        # Fraction of tokens dispatched to expert e (detached/stop-gradient)
        f_e = all_mask.mean(dim=0).detach()     # (E,)
        # Mean probability assigned to expert e
        P_e = all_G.mean(dim=0)                # (E,)

        loss = self.lambda_balance * float(n_experts) * torch.sum(f_e * P_e)
        return loss, float(loss.item())


class DomainConditionalLoadBalanceLoss(nn.Module):
    """
    Domain-Conditional Load Balancing Loss:
    Decouples load balancing per domain/task d to prevent a global penalty
    from forcing domain-specialized gates back into an identical uniform distribution.

    L_cond = lambda * sum_d w_d * (E * sum_e f_e^(d) * P_e^(d))
    """

    def __init__(self, lambda_balance: float = 0.02):
        super().__init__()
        self.lambda_balance = lambda_balance

    def forward(
        self,
        task_G_dict: Dict[str, torch.Tensor],
        task_mask_dict: Dict[str, torch.Tensor],
        n_experts: int,
        task_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            task_G_dict: Dict mapping task_name -> G tensor (B_t, E)
            task_mask_dict: Dict mapping task_name -> mask tensor (B_t, E)
            n_experts: Number of routed experts E
            task_weights: Optional per-task loss weighting
        """
        if not task_G_dict or self.lambda_balance <= 0.0:
            device = next(iter(task_G_dict.values())).device if task_G_dict else "cpu"
            return torch.tensor(0.0, device=device), 0.0

        total_loss = torch.tensor(0.0, device=next(iter(task_G_dict.values())).device)
        n_tasks = len(task_G_dict)

        for task, G in task_G_dict.items():
            mask = task_mask_dict[task]
            f_e = mask.mean(dim=0).detach()  # (E,)
            P_e = G.mean(dim=0)             # (E,)

            task_weight = task_weights.get(task, 1.0 / n_tasks) if task_weights else (1.0 / n_tasks)
            task_bal = float(n_experts) * torch.sum(f_e * P_e)
            total_loss = total_loss + task_weight * task_bal

        final_loss = self.lambda_balance * total_loss
        return final_loss, float(final_loss.item())


class DynamicBiasTracker:
    """
    Auxiliary-Loss-Free Dynamic Router Bias (DeepSeek-V3 inspired):
    Maintains an exponential moving average (EMA) of expert load and computes
    dynamic affine biases gamma_e:
        gamma_e = gamma_e + sign(target_load - current_load) * step_size
    This balances expert load dynamically at runtime without adding auxiliary gradients
    to the task loss, eliminating loss interference.
    """

    def __init__(
        self,
        n_experts: int,
        target_load: Optional[float] = None,
        decay: float = 0.9,
        step_size: float = 0.01,
        device: str = "cpu",
    ):
        self.n_experts = n_experts
        self.target_load = target_load if target_load is not None else (1.0 / n_experts)
        self.decay = decay
        self.step_size = step_size
        self.device = device
        self.ema_load = torch.ones(n_experts, device=device) * self.target_load
        self.dynamic_bias = torch.zeros(n_experts, device=device)

    def update(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Update EMA load counters and adjust dynamic biases.
        Args:
            mask: Binary selection mask (B, E)
        Returns:
            Current dynamic bias tensor (E,)
        """
        current_load = mask.float().mean(dim=0).detach()
        self.ema_load = self.decay * self.ema_load + (1 - self.decay) * current_load

        # Adjust dynamic bias in direction of under-utilized experts
        load_diff = self.target_load - self.ema_load
        self.dynamic_bias = self.dynamic_bias + self.step_size * torch.sign(load_diff)
        return self.dynamic_bias
