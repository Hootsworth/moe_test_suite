import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from moe_framework.models import Expert, ExpertPool, TaskGate, MultiGateMoEClassifier
from moe_framework.losses import (
    StandardLoadBalanceLoss,
    DomainConditionalLoadBalanceLoss,
    DynamicBiasTracker,
)
from moe_framework.data import SyntheticMoEWorld, create_task_dataloaders
from moe_framework.metrics import (
    compute_gini,
    cosine_similarity,
    compute_gate_statistics,
    compute_gate_cosine_similarity,
    compute_gini_specialization,
    evaluate_hypothesis,
)


def test_expert_pool():
    B, D, H, E = 8, 16, 32, 5
    pool = ExpertPool(n_experts=E, in_dim=D, hidden_dim=H, out_dim=D, activation="gelu")
    x = torch.randn(B, D)
    outs = pool(x)
    assert outs.shape == (B, E, D), f"Expected (B, E, D) but got {outs.shape}"


def test_task_gate_multigate():
    B, D, E, k = 10, 20, 6, 2
    gate = TaskGate(in_dim=D, n_experts=E, gate_type="multi_gate", tasks=["t1", "t2"])
    x = torch.randn(B, D)
    Gm, G, indices, mask = gate(x, task="t1", top_k=k)

    assert Gm.shape == (B, E)
    assert G.shape == (B, E)
    assert indices.shape == (B, k)
    assert mask.shape == (B, E)
    assert torch.all(mask.sum(dim=-1) == k)


def test_continuous_acoustic_gate():
    B, D, D_ac, E, k = 6, 20, 4, 5, 2
    gate = TaskGate(in_dim=D, n_experts=E, gate_type="continuous_acoustic", acoustic_dim=D_ac)
    x = torch.randn(B, D)
    z = torch.randn(B, D_ac)
    Gm, G, indices, mask = gate(x, acoustic_emb=z, top_k=k)

    assert Gm.shape == (B, E)
    assert mask.sum(dim=-1).int().tolist() == [k] * B


def test_shared_plus_routed_classifier():
    B, D, K, E_r, E_s, k = 12, 16, 5, 4, 2, 2
    model = MultiGateMoEClassifier(
        in_dim=D,
        n_classes=K,
        n_routed_experts=E_r,
        n_shared_experts=E_s,
        top_k=k,
        hidden_dim=24,
        gate_type="multi_gate",
        tasks=["t1", "t2"],
    )
    x = torch.randn(B, D)
    out = model(x, task="t1")

    assert out["logits"].shape == (B, K)
    assert out["Gm"].shape == (B, E_r)
    assert out["mask"].shape == (B, E_r)


def test_ablation_masking():
    B, D, K, E_r = 4, 16, 5, 4
    model = MultiGateMoEClassifier(
        in_dim=D,
        n_classes=K,
        n_routed_experts=E_r,
        top_k=2,
        gate_type="single_gate",
    )
    x = torch.randn(B, D)
    out = model(x, ablate_expert_ids=[1, 3])
    # Expert 1 and 3 should have 0 dispatch weights
    assert (out["Gm"][:, 1] == 0).all()
    assert (out["Gm"][:, 3] == 0).all()


def test_domain_conditional_load_balance_loss():
    E = 4
    G1, G2 = torch.softmax(torch.randn(10, E), dim=-1), torch.softmax(torch.randn(10, E), dim=-1)
    mask1, mask2 = torch.zeros(10, E), torch.zeros(10, E)
    mask1[:, :2] = 1.0
    mask2[:, 2:] = 1.0

    loss_fn = DomainConditionalLoadBalanceLoss(lambda_balance=0.05)
    loss, val = loss_fn({"d1": G1, "d2": G2}, {"d1": mask1, "d2": mask2}, n_experts=E)

    assert loss.ndim == 0
    assert val > 0


def test_gini_and_metrics():
    # High specialization: expert only active on task 1
    v_skewed = np.array([1.0, 0.0, 0.0])
    gini_val = compute_gini(v_skewed)
    assert gini_val > 0.5, f"Expected high Gini for skewed vector, got {gini_val}"

    # Uniform usage: equal across all tasks
    v_uniform = np.array([1.0, 1.0, 1.0])
    gini_uni = compute_gini(v_uniform)
    assert gini_uni < 0.1, f"Expected near-zero Gini for uniform vector, got {gini_uni}"


if __name__ == "__main__":
    test_expert_pool()
    test_task_gate_multigate()
    test_continuous_acoustic_gate()
    test_shared_plus_routed_classifier()
    test_ablation_masking()
    test_domain_conditional_load_balance_loss()
    test_gini_and_metrics()
    print("All unit tests passed successfully!")
