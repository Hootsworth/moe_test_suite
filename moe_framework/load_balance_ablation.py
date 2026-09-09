"""
Load Balancing Loss Sensitivity Study (lambda_bal in [0.0, 0.005, 0.015, 0.05])
"""

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from moe_framework.hybrid_moe import HybridSequenceMoE
from moe_framework.losses import DomainConditionalLoadBalanceLoss
from moe_framework.production_speech_pipeline import ProductionSpeechWorld, create_production_dataloaders
from moe_framework.ctc_engine import SequenceErrorEvaluator, ctc_greedy_decode

DOMAINS = ["adult_speech", "child_speech", "dysarthric_speech"]


def run_lambda_study(lambdas: List[float] = [0.0, 0.005, 0.015, 0.05], n_seeds: int = 5, out_dir: str = "results_rigorous_20seed"):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []

    for lam in lambdas:
        for s in range(1, n_seeds + 1):
            cfg = {
                "in_dim": 80, "acoustic_dim": 4, "n_routed_experts": 4, "n_shared_experts": 1,
                "top_k": 2, "hidden_dim": 48, "epochs": 20, "batch_size": 25, "lr": 0.008,
                "n_train_per_domain": 200, "n_val_per_domain": 40, "n_test_per_domain": 50,
                "seed": s, "lambda_balance": lam,
            }
            torch.manual_seed(s)
            np.random.seed(s)

            world = ProductionSpeechWorld(cfg)
            tokenizer = world.tokenizer
            train_loaders, _, test_loaders = create_production_dataloaders(world, DOMAINS, cfg, cfg["batch_size"])

            model = HybridSequenceMoE(
                in_dim=80, n_vocab=tokenizer.vocab_size - 1, n_routed_experts=4, n_shared_experts=1,
                top_k=2, hidden_dim=48, gate_type="multi_gate", tasks=DOMAINS,
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
            ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
            bal_loss_fn = DomainConditionalLoadBalanceLoss(lambda_balance=lam) if lam > 0 else None

            # Train
            model.train()
            min_batches = min(len(train_loaders[d]) for d in DOMAINS)
            for epoch in range(cfg["epochs"]):
                iterators = {d: iter(train_loaders[d]) for d in DOMAINS}
                for _ in range(min_batches):
                    optimizer.zero_grad()
                    task_G_dict, task_mask_dict = {}, {}
                    tot_ctc = torch.tensor(0.0, device=device)
                    for domain in DOMAINS:
                        b = next(iterators[domain])
                        out = model(b["x"].to(device), b["input_lengths"].to(device), task=domain)
                        loss = ctc_loss_fn(out["log_probs_ctc"], b["targets"].to(device), out["sub_lengths"], b["target_lengths"].to(device))
                        tot_ctc = tot_ctc + loss
                        task_G_dict[domain] = out["G"]
                        task_mask_dict[domain] = out["mask"]

                    bal = torch.tensor(0.0, device=device)
                    if bal_loss_fn is not None:
                        bal, _ = bal_loss_fn(task_G_dict, task_mask_dict, 4)
                    (tot_ctc + bal).backward()
                    optimizer.step()

            # Test & Load Entropy
            model.eval()
            with torch.no_grad():
                all_masks = []
                for domain in DOMAINS:
                    evaluator = SequenceErrorEvaluator(vocab_map=tokenizer.id_to_char, space_token=tokenizer.space_id)
                    for b in test_loaders[domain]:
                        out = model(b["x"].to(device), b["input_lengths"].to(device), task=domain)
                        hyp_tokens = ctc_greedy_decode(out["logits"], out["sub_lengths"], blank_idx=0)
                        ref_tokens = [b["targets"][i, :int(b["target_lengths"][i])].tolist() for i in range(b["targets"].size(0))]
                        evaluator.update(ref_tokens, hyp_tokens)
                        all_masks.append(out["mask"].cpu().numpy())

                    m = evaluator.compute()
                    records.append({
                        "lambda_bal": lam,
                        "seed": s,
                        "domain": domain,
                        "WER": m["WER"],
                    })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "load_balance_sensitivity.csv"), index=False)
    piv = df.groupby(["lambda_bal", "domain"])["WER"].agg(["mean", "std"]).round(2).reset_index()
    print("=== LOAD BALANCE SENSITIVITY (MEAN ± STD WER) ===")
    print(piv.to_string(index=False))
    return piv


if __name__ == "__main__":
    run_lambda_study()
