"""
Unit tests for Speech Production Pipeline and Hybrid MoE Architecture
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np

from moe_framework.production_speech_pipeline import (
    EnglishPhoneticTokenizer,
    ProductionSpeechWorld,
    create_production_dataloaders,
)
from moe_framework.hybrid_moe import HybridSequenceMoE


def test_disjointness():
    cfg = {"seed": 42}
    world = ProductionSpeechWorld(cfg)

    # Mathematical disjointness check
    train_set = set(world.TRAIN_SENTENCES)
    val_set = set(world.VAL_SENTENCES)
    overlap = train_set.intersection(val_set)

    assert len(overlap) == 0, f"Disjointness violated! Overlapping sentences: {overlap}"
    assert len(train_set) > 0 and len(val_set) > 0


def test_tokenizer():
    tokenizer = EnglishPhoneticTokenizer()
    text = "THE QUICK BROWN FOX JUMPS"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert len(tokens) > 0
    assert decoded == text
    assert tokenizer.blank_id == 0


def test_production_dataloaders():
    cfg = {"seed": 42, "n_train_per_domain": 10, "n_val_per_domain": 5}
    world = ProductionSpeechWorld(cfg)
    domains = ["adult_speech", "child_speech", "dysarthric_speech"]

    train_loaders, val_loaders = create_production_dataloaders(world, domains, cfg, batch_size=4)
    batch = next(iter(train_loaders["child_speech"]))

    assert batch["x"].size(2) == 80
    assert batch["targets"].ndim == 2
    assert batch["input_lengths"].size(0) == 4
    assert batch["z_acoustic"].size(1) == 4


def test_hybrid_moe_forward_and_loss():
    tokenizer = EnglishPhoneticTokenizer()
    model = HybridSequenceMoE(
        in_dim=80,
        n_vocab=tokenizer.vocab_size - 1,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        hidden_dim=48,
        gate_type="multi_gate",
        tasks=["adult_speech", "child_speech", "dysarthric_speech"],
    )

    B, T_raw = 4, 100
    x = torch.randn(B, T_raw, 80)
    in_lens = torch.full((B,), T_raw, dtype=torch.long)
    targets = torch.randint(1, tokenizer.vocab_size, size=(B, 6))
    tar_lens = torch.full((B,), 6, dtype=torch.long)

    out = model(x=x, input_lengths=in_lens, task="dysarthric_speech")
    log_probs = out["log_probs_ctc"]
    sub_lens = out["sub_lengths"]

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = ctc_loss_fn(log_probs, targets, sub_lens, tar_lens)

    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()

    # Verify gradients flowed into shared expert
    for ex in model.shared_pool.experts:
        assert ex.net[0].weight.grad is not None
    # Verify gradients flowed into routed experts
    for ex in model.routed_pool.experts:
        assert ex.net[0].weight.grad is not None


if __name__ == "__main__":
    test_disjointness()
    test_tokenizer()
    test_production_dataloaders()
    test_hybrid_moe_forward_and_loss()
    print("All Speech Production and Hybrid MoE unit tests passed successfully!")
