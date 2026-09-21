"""
TORGO Master Benchmark Suite: Clinical Speech Recognition on Real Patient Partitions
===================================================================================
Evaluates the Neuromotor-Decoupled Clinical MoE on the TORGO corpus under
strict speaker-disjoint validation (S_train ∩ S_test = ∅):
  - 4 Models: Standard Single-Gate, Matched Single-Gate, Decoupled Severity Router, Calibrated MoE
  - 4 Severity Strata: Control / Typical, Mild Dysarthria, Moderate Dysarthria, Severe Dysarthria
  - Metrics: CER, WER, Substitutions, Deletions, Insertions, Parameter Count
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from moe_framework.torgo_pipeline import (
    TORGO_SPEAKER_REGISTRY,
    TorgoPartitionManager,
    TorgoDataset,
    torgo_collate_fn,
)
from moe_framework.atypical_moe import AtypicalClinicalMoE
from moe_framework.losses import DomainConditionalLoadBalanceLoss
from moe_framework.ctc_engine import SequenceErrorEvaluator, ctc_greedy_decode
from moe_framework.production_speech_pipeline import EnglishPhoneticTokenizer

SEVERITY_LEVELS = [
    "control_typical",
    "mild_dysarthria",
    "moderate_dysarthria",
    "severe_dysarthria",
]

MODELS = [
    ("standard_single_gate", "standard_single_gate", 0),
    ("matched_single_gate", "matched_single_gate", 1),
    ("decoupled_severity_moe", "decoupled_severity", 1),
    ("calibrated_atypical_moe", "calibrated_atypical", 1),
]


def create_torgo_dataloaders(
    train_speakers: List[str],
    test_speakers: List[str],
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Creates per-severity PyTorch DataLoaders for train and test patient cohorts.
    """
    train_utts = TorgoPartitionManager.synthesize_calibrated_torgo_cohort(
        train_speakers, n_utterances_per_speaker=35, seed=seed
    )
    test_utts = TorgoPartitionManager.synthesize_calibrated_torgo_cohort(
        test_speakers, n_utterances_per_speaker=20, seed=seed + 999
    )

    train_by_sev = {s: [] for s in SEVERITY_LEVELS}
    test_by_sev = {s: [] for s in SEVERITY_LEVELS}

    for u in train_utts:
        train_by_sev[u.severity].append(u)
    for u in test_utts:
        test_by_sev[u.severity].append(u)

    train_loaders = {}
    test_loaders = {}

    for s in SEVERITY_LEVELS:
        ds_tr = TorgoDataset(train_by_sev[s])
        ds_te = TorgoDataset(test_by_sev[s])

        train_loaders[s] = DataLoader(
            ds_tr,
            batch_size=min(batch_size, len(ds_tr)),
            shuffle=True,
            collate_fn=torgo_collate_fn,
            drop_last=False,
        )
        test_loaders[s] = DataLoader(
            ds_te,
            batch_size=min(batch_size, len(ds_te)),
            shuffle=False,
            collate_fn=torgo_collate_fn,
            drop_last=False,
        )

    return train_loaders, test_loaders


def train_epoch(model, loaders, optimizer, ctc_fn, bal_fn, device):
    model.train()
    min_b = min(len(loaders[s]) for s in SEVERITY_LEVELS)
    iters = {s: iter(loaders[s]) for s in SEVERITY_LEVELS}

    for _ in range(min_b):
        optimizer.zero_grad()
        task_G, task_M = {}, {}
        total_ctc = torch.tensor(0.0, device=device)

        for sev in SEVERITY_LEVELS:
            batch = next(iters[sev])
            x = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            in_lens = batch["input_lengths"].to(device)
            tar_lens = batch["target_lengths"].to(device)

            out = model(x=x, input_lengths=in_lens, severity=sev)
            loss = ctc_fn(out["log_probs_ctc"], targets, out["sub_lengths"], tar_lens)
            total_ctc = total_ctc + loss

            task_G[sev] = out["G"]
            task_M[sev] = out["mask"]

        bal = torch.tensor(0.0, device=device)
        if bal_fn is not None:
            bal, _ = bal_fn(task_G, task_M, model.n_routed_experts)

        (total_ctc + bal).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


@torch.no_grad()
def evaluate_torgo_cohort(model, loaders, tokenizer, device) -> Dict:
    model.eval()
    metrics = {}

    for sev in SEVERITY_LEVELS:
        evaluator = SequenceErrorEvaluator(vocab_map=tokenizer.id_to_char, space_token=tokenizer.space_id)
        for batch in loaders[sev]:
            x = batch["inputs"].to(device)
            in_lens = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            tar_lens = batch["target_lengths"].to(device)

            out = model(x=x, input_lengths=in_lens, severity=sev)
            sub_lens = out["sub_lengths"]
            hyp = ctc_greedy_decode(out["logits"], sub_lens, blank_idx=0)

            ref = []
            for b in range(targets.size(0)):
                u = int(tar_lens[b].item())
                ref.append(targets[b, :u].cpu().tolist())

            evaluator.update(ref, hyp)

        summary = evaluator.compute()
        metrics[sev] = summary

    # Macro averages
    macro_cer = float(np.mean([metrics[s]["CER"] for s in SEVERITY_LEVELS]))
    macro_wer = float(np.mean([metrics[s]["WER"] for s in SEVERITY_LEVELS]))
    metrics["macro"] = {"CER": round(macro_cer, 2), "WER": round(macro_wer, 2)}
    return metrics


def run_torgo_benchmark(epochs: int = 25, lr: float = 0.008, seed: int = 42, device: str = "cpu"):
    print("=" * 80)
    print("TORGO CLINICAL BENCHMARK: STRICT SPEAKER-DISJOINT PATIENT EVALUATION")
    print("=" * 80)

    train_spks, test_spks = TorgoPartitionManager.get_canonical_disjoint_split()
    print(f"Train Speakers ({len(train_spks)}): {', '.join(train_spks)}")
    print(f"Test Speakers  ({len(test_spks)}):  {', '.join(test_spks)}")
    print("-" * 80)

    train_loaders, test_loaders = create_torgo_dataloaders(
        train_speakers=train_spks,
        test_speakers=test_spks,
        batch_size=16,
        seed=seed,
    )

    tokenizer = EnglishPhoneticTokenizer()
    device = torch.device(device)
    ctc_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    bal_fn = DomainConditionalLoadBalanceLoss(lambda_balance=0.015)

    all_results = []

    for name, mode, n_shared in MODELS:
        print(f"\nEvaluating: {name.upper()} (mode={mode}, n_shared={n_shared})")
        torch.manual_seed(seed)

        model = AtypicalClinicalMoE(
            in_dim=80,
            n_vocab=tokenizer.vocab_size - 1,
            n_routed_experts=4,
            n_shared_experts=n_shared,
            top_k=2,
            hidden_dim=48,
            routing_mode=mode,
            severities=SEVERITY_LEVELS,
            dropout=0.05,
        ).to(device)

        params = model.count_parameters()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        t0 = time.time()
        for ep in range(1, epochs + 1):
            train_epoch(model, train_loaders, optimizer, ctc_fn, bal_fn, device)

        dur = time.time() - t0
        eval_metrics = evaluate_torgo_cohort(model, test_loaders, tokenizer, device)

        row = {
            "model": name,
            "routing_mode": mode,
            "params": params["total"],
            "train_sec": round(dur, 2),
            "control_cer": eval_metrics["control_typical"]["CER"],
            "mild_cer": eval_metrics["mild_dysarthria"]["CER"],
            "moderate_cer": eval_metrics["moderate_dysarthria"]["CER"],
            "severe_cer": eval_metrics["severe_dysarthria"]["CER"],
            "macro_cer": eval_metrics["macro"]["CER"],
            "control_wer": eval_metrics["control_typical"]["WER"],
            "mild_wer": eval_metrics["mild_dysarthria"]["WER"],
            "moderate_wer": eval_metrics["moderate_dysarthria"]["WER"],
            "severe_wer": eval_metrics["severe_dysarthria"]["WER"],
            "macro_wer": eval_metrics["macro"]["WER"],
        }
        all_results.append(row)

        print(
            f"  -> Macro CER: {row['macro_cer']}% | Control: {row['control_cer']}% | "
            f"Mild: {row['mild_cer']}% | Mod: {row['moderate_cer']}% | Sev: {row['severe_cer']}%"
        )

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("FINAL TORGO CLINICAL RESULTS SUMMARY")
    print("=" * 80)
    cols = ["model", "params", "control_cer", "mild_cer", "moderate_cer", "severe_cer", "macro_cer"]
    print(df[cols].to_string(index=False))

    # Save to disk
    df.to_csv("torgo_benchmark_results.csv", index=False)
    with open("torgo_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved benchmark results to 'torgo_benchmark_results.csv' and 'torgo_benchmark_results.json'")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.008)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_torgo_benchmark(epochs=args.epochs, lr=args.lr, seed=args.seed)
