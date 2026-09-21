"""
Master Benchmark Suite for Atypical and Dysarthric Speech Recognition
====================================================================
Evaluates on strictly speaker-disjoint clinical patient cohorts (S_train ∩ S_test = ∅):
  1. Standard Single-Gate MoE (Baseline)
  2. Parameter-Matched Single-Gate MoE (Baseline)
  3. Decoupled Severity-Owned MoE (Ours)
  4. Calibrated Atypical MoE with Speaker Acoustic-Average Calibration (Ours)
"""

import argparse
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim

from moe_framework.atypical_clinical_pipeline import (
    ClinicalSpeechWorld,
    create_clinical_dataloaders,
    SEVERITY_LEVELS,
)
from moe_framework.atypical_moe import AtypicalClinicalMoE
from moe_framework.losses import DomainConditionalLoadBalanceLoss
from moe_framework.ctc_engine import SequenceErrorEvaluator, ctc_greedy_decode

CONFIG = {
    "in_dim": 80,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "top_k": 2,
    "hidden_dim": 48,
    "epochs": 25,
    "batch_size": 20,
    "lr": 0.008,
    "lambda_balance": 0.015,
    "n_train_per_patient": 45,
    "n_test_per_patient": 25,
}

MODELS = [
    ("standard_single_gate", "standard_single_gate", 0),
    ("matched_single_gate", "matched_single_gate", 1),
    ("decoupled_severity_moe", "decoupled_severity", 1),
    ("calibrated_atypical_moe", "calibrated_atypical", 1),
]


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
            x = batch["x"].to(device)
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
def evaluate_cohort(model, loaders, tokenizer, device):
    model.eval()
    metrics = {}
    expert_allocations = {}

    for sev in SEVERITY_LEVELS:
        evaluator = SequenceErrorEvaluator(vocab_map=tokenizer.id_to_char, space_token=tokenizer.space_id)
        all_masks = []

        for batch in loaders[sev]:
            x = batch["x"].to(device)
            targets = batch["targets"].to(device)
            in_lens = batch["input_lengths"].to(device)
            tar_lens = batch["target_lengths"].to(device)

            out = model(x=x, input_lengths=in_lens, severity=sev)
            hyp = ctc_greedy_decode(out["logits"], out["sub_lengths"], blank_idx=0)

            ref = []
            for b in range(targets.size(0)):
                u = int(tar_lens[b].item())
                ref.append(targets[b, :u].cpu().tolist())

            evaluator.update(ref, hyp)
            all_masks.append(out["mask"].cpu().numpy())

        metrics[sev] = evaluator.compute()
        all_M = np.concatenate(all_masks, axis=0)
        expert_allocations[sev] = all_M.mean(axis=0)

    return metrics, expert_allocations


def run_clinical_benchmark(n_seeds: int = 5, out_dir: str = "results_clinical_atypical"):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = []
    alloc_records = []

    print("==========================================================================================")
    print(f"   STARTING CLINICAL ATYPICAL SPEECH BENCHMARK (SPEAKER-DISJOINT, {n_seeds} SEEDS)      ")
    print("==========================================================================================")

    for s in range(1, n_seeds + 1):
        torch.manual_seed(s)
        np.random.seed(s)

        world = ClinicalSpeechWorld(seed=s)
        tokenizer = world.tokenizer
        train_loaders, test_loaders = create_clinical_dataloaders(
            world=world,
            batch_size=CONFIG["batch_size"],
            n_train_per_patient=CONFIG["n_train_per_patient"],
            n_test_per_patient=CONFIG["n_test_per_patient"],
        )

        for model_name, routing_mode, n_shared in MODELS:
            model = AtypicalClinicalMoE(
                in_dim=CONFIG["in_dim"],
                n_vocab=tokenizer.vocab_size - 1,
                n_routed_experts=CONFIG["n_routed_experts"],
                n_shared_experts=n_shared,
                top_k=CONFIG["top_k"],
                hidden_dim=CONFIG["hidden_dim"],
                routing_mode=routing_mode,
                severities=SEVERITY_LEVELS,
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
            ctc_fn = nn.CTCLoss(blank=0, zero_infinity=True)
            bal_fn = DomainConditionalLoadBalanceLoss(lambda_balance=CONFIG["lambda_balance"])

            for ep in range(CONFIG["epochs"]):
                train_epoch(model, train_loaders, optimizer, ctc_fn, bal_fn, device)

            # Evaluate on unseen held-out patients
            metrics, allocs = evaluate_cohort(model, test_loaders, tokenizer, device)

            for sev in SEVERITY_LEVELS:
                m = metrics[sev]
                records.append({
                    "seed": s,
                    "model_name": model_name,
                    "severity": sev,
                    "WER (%)": m["WER"],
                    "CER (%)": m["CER"],
                    "sub_rate": m["word_sub_rate"],
                    "del_rate": m["word_del_rate"],
                    "ins_rate": m["word_ins_rate"],
                })

                for e_idx, e_val in enumerate(allocs[sev]):
                    alloc_records.append({
                        "seed": s,
                        "model_name": model_name,
                        "severity": sev,
                        "expert_id": e_idx,
                        "dispatch_rate": float(e_val),
                    })

        print(f"Seed {s}/{n_seeds} completed.")

    df_res = pd.DataFrame(records)
    df_alloc = pd.DataFrame(alloc_records)

    df_res.to_csv(os.path.join(out_dir, "clinical_wer_cer_summary.csv"), index=False)
    df_alloc.to_csv(os.path.join(out_dir, "clinical_expert_allocations.csv"), index=False)

    print("\n==========================================================================================")
    print("      UNSEEN PATIENT COHORT TEST PERFORMANCE (MEAN ± STD ACROSS SEEDS)                   ")
    print("==========================================================================================")
    summary_piv = df_res.groupby(["model_name", "severity"])[["WER (%)", "CER (%)", "sub_rate", "del_rate"]].agg(["mean", "std"]).round(2)
    print(summary_piv.to_string())

    # Paired Significance Tests (Calibrated Atypical MoE vs. Baselines)
    print("\n==========================================================================================")
    print("      PAIRED STATISTICAL SIGNIFICANCE (CALIBRATED ATYPICAL MOE vs. BASELINES)            ")
    print("==========================================================================================")
    stat_rows = []
    for sev in SEVERITY_LEVELS:
        calib_wers = df_res[(df_res["model_name"] == "calibrated_atypical_moe") & (df_res["severity"] == sev)].sort_values("seed")["WER (%)"].values

        for base in ["standard_single_gate", "matched_single_gate", "decoupled_severity_moe"]:
            base_wers = df_res[(df_res["model_name"] == base) & (df_res["severity"] == sev)].sort_values("seed")["WER (%)"].values
            t_stat, t_pval = stats.ttest_rel(base_wers, calib_wers)
            calib_m = float(np.mean(calib_wers))
            base_m = float(np.mean(base_wers))
            rel_reduc = float((base_m - calib_m) / base_m * 100.0)

            stat_rows.append({
                "severity": sev,
                "comparison": f"calibrated_atypical vs {base}",
                "calibrated_wer": round(calib_m, 2),
                "baseline_wer": round(base_m, 2),
                "rel_reduction (%)": round(rel_reduc, 2),
                "p_value": f"{t_pval:.4e}",
            })

    df_stat = pd.DataFrame(stat_rows)
    df_stat.to_csv(os.path.join(out_dir, "clinical_statistical_significance.csv"), index=False)
    print(df_stat.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="results_clinical_atypical")
    args = parser.parse_args()

    run_clinical_benchmark(n_seeds=args.seeds, out_dir=args.out)
