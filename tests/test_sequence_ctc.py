"""
Unit tests for Sequence CTC and WER / CER Engine
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np

from moe_framework.ctc_engine import levenshtein_distance, ctc_greedy_decode, SequenceErrorEvaluator
from moe_framework.sequence_data import SpeechSequenceWorld, create_sequence_dataloaders
from moe_framework.sequence_models import SequenceMultiGateMoE


def test_levenshtein():
    # 1. Exact match
    r1 = levenshtein_distance([1, 2, 3], [1, 2, 3])
    assert r1["distance"] == 0 and r1["substitutions"] == 0 and r1["deletions"] == 0 and r1["insertions"] == 0

    # 2. Substitution: [1, 2, 3] -> [1, 5, 3]
    r2 = levenshtein_distance([1, 2, 3], [1, 5, 3])
    assert r2["distance"] == 1 and r2["substitutions"] == 1

    # 3. Deletion: [1, 2, 3] -> [1, 3]
    r3 = levenshtein_distance([1, 2, 3], [1, 3])
    assert r3["distance"] == 1 and r3["deletions"] == 1

    # 4. Insertion: [1, 2, 3] -> [1, 2, 4, 3]
    r4 = levenshtein_distance([1, 2, 3], [1, 2, 4, 3])
    assert r4["distance"] == 1 and r4["insertions"] == 1


def test_ctc_decode():
    # [blank, 1, 1, blank, 2, 2, 2, blank, 3] -> [1, 2, 3]
    seq = [0, 1, 1, 0, 2, 2, 2, 0, 3]
    logits = torch.zeros(1, len(seq), 5)
    for t, tok in enumerate(seq):
        logits[0, t, tok] = 10.0  # high logit for selected token

    decoded = ctc_greedy_decode(logits, blank_idx=0)
    assert decoded[0] == [1, 2, 3], f"Expected [1, 2, 3] but got {decoded[0]}"


def test_sequence_moe_forward_and_ctc_loss():
    B, T_raw, D, V = 4, 24, 20, 12
    model = SequenceMultiGateMoE(
        in_dim=D,
        n_vocab=V,
        n_routed_experts=4,
        top_k=2,
        hidden_dim=32,
        gate_type="multi_gate",
        tasks=["language", "child", "atypical"],
    )

    x = torch.randn(B, T_raw, D)
    targets = torch.randint(1, V + 1, size=(B, 5))
    input_lengths = torch.full((B,), T_raw, dtype=torch.long)
    target_lengths = torch.full((B,), 5, dtype=torch.long)

    out = model(x, input_lengths=input_lengths, task="child")
    log_probs = out["log_probs_ctc"]  # (T_sub, B, V+1)
    sub_lengths = out["sub_lengths"]  # (B,)

    assert log_probs.size(1) == B
    assert sub_lengths.size(0) == B

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = ctc_loss_fn(log_probs, targets, sub_lengths, target_lengths)

    assert loss.ndim == 0
    assert not torch.isnan(loss)
    loss.backward()

    # Verify gradients flowed into experts
    for ex in model.routed_pool.experts:
        assert ex.net[0].weight.grad is not None


def test_sequence_evaluator():
    evaluator = SequenceErrorEvaluator()
    ref = [[1, 2, 3], [4, 5]]
    hyp = [[1, 2, 3], [4, 6]]  # 1 substitution out of 5 chars
    evaluator.update(ref, hyp)
    res = evaluator.compute()

    assert res["CER"] == 20.0  # 1 / 5 = 20%
    assert res["total_chars"] == 5


if __name__ == "__main__":
    test_levenshtein()
    test_ctc_decode()
    test_sequence_moe_forward_and_ctc_loss()
    test_sequence_evaluator()
    print("All Sequence CTC and WER unit tests passed successfully!")
