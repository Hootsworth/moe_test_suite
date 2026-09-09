"""
PyTorch Multi-Gate Mixture of Experts (MoE) Scaling Research Suite
==================================================================
Comprehensive multi-task MoE benchmark implementing:
  - Multi-gate routing (Task-ID conditioned)
  - Continuous acoustic gating (Zero-shot acoustic vector routing)
  - Single-gate baseline
  - DeepSeek-style Shared + Routed expert hybrid architecture
  - Domain-conditional vs Standard load balancing vs Dynamic bias
  - Gini Specialization Index & Mechanistic Expert Ablation Analysis
  - Multi-seed sensitivity evaluation
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from moe_framework.data import SyntheticMoEWorld, create_task_dataloaders
from moe_framework.losses import (
    DomainConditionalLoadBalanceLoss,
    DynamicBiasTracker,
    StandardLoadBalanceLoss,
)
from moe_framework.metrics import (
    compute_gate_cosine_similarity,
    compute_gate_statistics,
    compute_gini_specialization,
    evaluate_hypothesis,
    run_expert_ablation,
)
from moe_framework.models import MultiGateMoEClassifier

TASKS = ["language", "child", "atypical"]

DEFAULT_CONFIG = {
    "seed": 7,
    "n_classes": 12,
    "dim_a": 10,
    "dim_b": 10,
    "acoustic_dim": 4,
    "n_routed_experts": 5,
    "n_shared_experts": 0,
    "top_k": 2,
    "hidden_dim": 24,
    "activation": "relu",
    "dropout": 0.0,
    "renormalize_topk": False,
    "n_train_per_task": 800,
    "n_val_per_task": 400,
    "epochs": 35,
    "batch_size": 40,
    "lr": 0.01,
    "lambda_balance": 0.02,
    "balance_loss_type": "domain_conditional",  # 'standard', 'domain_conditional', 'dynamic_bias', 'none'
    "noise_std": 1.05,
    "class_scale": 1.4,
    "alpha_mild": 0.15,
    "alpha_strong": 0.8,
    "mag_child": 1.0,
    "mag_atypical": 1.25,
    "mag_language": 0.0,
}


def train_epoch(
    model: MultiGateMoEClassifier,
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    balance_loss_fn: Optional[nn.Module],
    dynamic_tracker: Optional[DynamicBiasTracker],
    cfg: Dict[str, Union[int, float, str]],
    device: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.train()
    task_losses = {t: [] for t in TASKS}
    task_accs = {t: [] for t in TASKS}

    # Zip batches across tasks
    iterators = {t: iter(train_loaders[t]) for t in TASKS}
    min_batches = min(len(loader) for loader in train_loaders.values())

    for _ in range(min_batches):
        optimizer.zero_grad()
        task_G_dict, task_mask_dict = {}, {}
        total_cls_loss = torch.tensor(0.0, device=device)

        dynamic_bias = None
        if dynamic_tracker is not None:
            dynamic_bias = dynamic_tracker.dynamic_bias

        for task in TASKS:
            batch = next(iterators[task])
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            z = batch["z_acoustic"].to(device)

            out = model(
                x=x,
                task=task,
                acoustic_emb=z,
                dynamic_bias=dynamic_bias,
            )

            cls_loss = criterion(out["logits"], y)
            total_cls_loss = total_cls_loss + cls_loss

            preds = out["logits"].argmax(dim=-1)
            acc = (preds == y).float().mean().item()

            task_losses[task].append(cls_loss.item())
            task_accs[task].append(acc)

            task_G_dict[task] = out["G"]
            task_mask_dict[task] = out["mask"]

            if dynamic_tracker is not None:
                dynamic_tracker.update(out["mask"])

        # Compute load balance loss
        bal_loss = torch.tensor(0.0, device=device)
        bal_type = cfg.get("balance_loss_type", "domain_conditional")

        if bal_type == "domain_conditional" and isinstance(balance_loss_fn, DomainConditionalLoadBalanceLoss):
            bal_loss, _ = balance_loss_fn(task_G_dict, task_mask_dict, model.n_routed_experts)
        elif bal_type == "standard" and isinstance(balance_loss_fn, StandardLoadBalanceLoss):
            bal_loss, _ = balance_loss_fn(
                list(task_G_dict.values()),
                list(task_mask_dict.values()),
                model.n_routed_experts,
            )

        total_step_loss = total_cls_loss + bal_loss
        total_step_loss.backward()
        optimizer.step()

    mean_losses = {t: float(np.mean(task_losses[t])) for t in TASKS}
    mean_accs = {t: float(np.mean(task_accs[t])) for t in TASKS}
    return mean_losses, mean_accs


@torch.no_grad()
def evaluate_model(
    model: MultiGateMoEClassifier,
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    criterion: nn.Module,
    device: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    val_losses = {}
    val_accs = {}

    for task in TASKS:
        losses, correct, total = [], 0, 0
        for batch in val_loaders[task]:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            z = batch["z_acoustic"].to(device)

            out = model(x=x, task=task, acoustic_emb=z)
            loss = criterion(out["logits"], y)
            losses.append(loss.item())

            preds = out["logits"].argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        val_losses[task] = float(np.mean(losses)) if losses else 0.0
        val_accs[task] = correct / total if total > 0 else 0.0

    return val_losses, val_accs


def train_single_model(
    gate_type: str,
    cfg: Dict[str, Union[int, float, str]],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
    curve_log: List[Dict[str, Union[str, int, float]]],
) -> MultiGateMoEClassifier:
    in_dim = int(cfg["dim_a"]) + int(cfg["dim_b"])
    model = MultiGateMoEClassifier(
        in_dim=in_dim,
        n_classes=int(cfg["n_classes"]),
        n_routed_experts=int(cfg["n_routed_experts"]),
        n_shared_experts=int(cfg.get("n_shared_experts", 0)),
        top_k=int(cfg["top_k"]),
        hidden_dim=int(cfg["hidden_dim"]),
        gate_type=gate_type,
        tasks=TASKS,
        acoustic_dim=int(cfg["acoustic_dim"]),
        activation=str(cfg.get("activation", "relu")),
        dropout=float(cfg.get("dropout", 0.0)),
        renormalize_topk=bool(cfg.get("renormalize_topk", False)),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = nn.CrossEntropyLoss()

    bal_type = cfg.get("balance_loss_type", "domain_conditional")
    if bal_type == "domain_conditional":
        bal_loss_fn = DomainConditionalLoadBalanceLoss(lambda_balance=float(cfg["lambda_balance"]))
        dynamic_tracker = None
    elif bal_type == "standard":
        bal_loss_fn = StandardLoadBalanceLoss(lambda_balance=float(cfg["lambda_balance"]))
        dynamic_tracker = None
    elif bal_type == "dynamic_bias":
        bal_loss_fn = None
        dynamic_tracker = DynamicBiasTracker(n_experts=int(cfg["n_routed_experts"]), device=device)
    else:
        bal_loss_fn = None
        dynamic_tracker = None

    for epoch in range(int(cfg["epochs"])):
        tr_losses, tr_accs = train_epoch(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            criterion=criterion,
            balance_loss_fn=bal_loss_fn,
            dynamic_tracker=dynamic_tracker,
            cfg=cfg,
            device=device,
        )
        va_losses, va_accs = evaluate_model(
            model=model,
            val_loaders=val_loaders,
            criterion=criterion,
            device=device,
        )

        for task in TASKS:
            curve_log.append({
                "model_gate_type": gate_type,
                "task": task,
                "epoch": epoch,
                "train_loss": round(tr_losses[task], 4),
                "train_acc": round(tr_accs[task], 4),
                "val_loss": round(va_losses[task], 4),
                "val_acc": round(va_accs[task], 4),
            })

    return model


def run_benchmark(
    out_dir: str = "results_torch",
    seed_override: Optional[int] = None,
    include_continuous: bool = True,
    include_ablation: bool = True,
    device: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if seed_override is not None:
        cfg["seed"] = seed_override

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    world = SyntheticMoEWorld(cfg)
    train_loaders, val_loaders = create_task_dataloaders(
        world=world,
        tasks=TASKS,
        cfg=cfg,
        batch_size=int(cfg["batch_size"]),
    )

    gate_models_to_test = ["multi_gate", "single_gate"]
    if include_continuous:
        gate_models_to_test.append("continuous_acoustic")

    curve_log = []
    trained_models = {}
    usage_records = []
    similarity_records = []
    gini_records = []
    ablation_records = []

    t0 = time.time()
    for gtype in gate_models_to_test:
        if not quiet:
            print(f"--> Training model with gate_type: '{gtype}' on {device}...")
        model = train_single_model(
            gate_type=gtype,
            cfg=cfg,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            device=device,
            curve_log=curve_log,
        )
        trained_models[gtype] = model

        # 1. Gate statistics
        stat_rows, gate_vecs = compute_gate_statistics(model, val_loaders, TASKS, device=device)
        usage_records.extend(stat_rows)

        # 2. Gate Cosine Similarity
        sim_rows = compute_gate_cosine_similarity(gate_vecs, model_gate_type=gtype)
        similarity_records.extend(sim_rows)

        # 3. Gini Specialization Index
        if gtype != "single_gate":
            gini_dict = compute_gini_specialization(gate_vecs)
            gini_dict["model_gate_type"] = gtype
            gini_records.append(gini_dict)

        # 4. Expert Ablation Analysis
        if include_ablation:
            abl_rows = run_expert_ablation(model, val_loaders, TASKS, device=device)
            for r in abl_rows:
                r["model_gate_type"] = gtype
            ablation_records.extend(abl_rows)

    elapsed = time.time() - t0

    # Save outputs to CSVs
    df_curves = pd.DataFrame(curve_log)
    df_usage = pd.DataFrame(usage_records)
    df_sim = pd.DataFrame(similarity_records)
    df_gini = pd.DataFrame(gini_records)
    df_ablation = pd.DataFrame(ablation_records) if ablation_records else pd.DataFrame()

    df_curves.to_csv(os.path.join(out_dir, "training_curves.csv"), index=False)
    df_usage.to_csv(os.path.join(out_dir, "expert_usage.csv"), index=False)
    df_sim.to_csv(os.path.join(out_dir, "gate_similarity.csv"), index=False)
    df_gini.to_csv(os.path.join(out_dir, "gini_specialization.csv"), index=False)
    if not df_ablation.empty:
        df_ablation.to_csv(os.path.join(out_dir, "expert_ablation.csv"), index=False)

    # Hypothesis Verification
    hyp_rows = []
    for gtype in ["multi_gate", "continuous_acoustic"]:
        sub_sim = [r for r in similarity_records if r["model_gate_type"] == gtype]
        if sub_sim:
            hyp_res = evaluate_hypothesis(sub_sim)
            hyp_res["model_gate_type"] = gtype
            hyp_rows.append(hyp_res)

    df_hyp = pd.DataFrame(hyp_rows)
    df_hyp.to_csv(os.path.join(out_dir, "hypothesis_check.csv"), index=False)

    # Summary performance table
    summary_rows = []
    for gtype in gate_models_to_test:
        sub = df_curves[df_curves["model_gate_type"] == gtype]
        final_ep = sub["epoch"].max()
        for task in TASKS:
            task_sub = sub[sub["task"] == task]
            final_row = task_sub[task_sub["epoch"] == final_ep].iloc[0]
            best_val = task_sub["val_acc"].max()
            summary_rows.append({
                "model_gate_type": gtype,
                "task": task,
                "final_train_loss": final_row["train_loss"],
                "final_train_acc": final_row["train_acc"],
                "final_val_loss": final_row["val_loss"],
                "final_val_acc": final_row["val_acc"],
                "best_val_acc": round(float(best_val), 4),
            })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Config record
    cfg_rows = [{"parameter": k, "value": str(v)} for k, v in cfg.items()]
    cfg_rows.append({"parameter": "elapsed_seconds", "value": str(round(elapsed, 2))})
    cfg_rows.append({"parameter": "device", "value": device})
    pd.DataFrame(cfg_rows).to_csv(os.path.join(out_dir, "config.csv"), index=False)

    if not quiet:
        print(f"\nTraining completed in {elapsed:.2f}s on {device}.")
        print("\n=== Validation Performance Summary ===")
        print(df_summary.to_string(index=False))
        print("\n=== Gate Cosine Similarity Matrix ===")
        print(df_sim.to_string(index=False))
        print("\n=== Hypothesis Check ===")
        print(df_hyp.to_string(index=False))
        if not df_gini.empty:
            print("\n=== Expert Gini Specialization Index ===")
            print(df_gini.to_string(index=False))

    return {
        "summary": df_summary,
        "hypothesis": df_hyp,
        "similarity": df_sim,
        "gini": df_gini,
        "ablation": df_ablation,
    }


def run_seed_sensitivity(
    seeds: List[int],
    base_out: str = "results_torch",
) -> pd.DataFrame:
    """Run sensitivity across multiple seeds in PyTorch to test statistical significance."""
    rows = []
    seed_dir = os.path.join(base_out, "seed_sensitivity")
    os.makedirs(seed_dir, exist_ok=True)

    print(f"\n[Multi-Seed Sensitivity Sweep] Testing {len(seeds)} random seeds: {seeds}...")
    for s in seeds:
        print(f"--> Running seed={s} ...")
        res = run_benchmark(
            out_dir=os.path.join(seed_dir, f"seed_{s}"),
            seed_override=s,
            include_continuous=True,
            include_ablation=False,
            quiet=True,
        )
        for _, hyp_row in res["hypothesis"].iterrows():
            record = dict(hyp_row)
            record["seed"] = s
            rows.append(record)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base_out, "hypothesis_check_across_seeds.csv"), index=False)

    print("\n=== Multi-Seed Statistical Hypothesis Results ===")
    print(df[["seed", "model_gate_type", "similarity_child_atypical", "avg_similarity_to_language", "margin", "hypothesis_supported"]].to_string(index=False))

    for gtype in ["multi_gate", "continuous_acoustic"]:
        sub = df[df["model_gate_type"] == gtype]
        if not sub.empty:
            hit_rate = sub["hypothesis_supported"].sum()
            print(f"[{gtype}] Hypothesis Supported: {hit_rate}/{len(sub)} ({hit_rate/len(sub)*100:.1f}%) | Mean Margin: {sub['margin'].mean():.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Multi-Gate MoE Scaling Suite")
    parser.add_argument("--out", default="results_torch", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda', 'mps')")
    parser.add_argument("--no-continuous", action="store_true", help="Skip continuous acoustic gate")
    parser.add_argument("--no-ablation", action="store_true", help="Skip expert ablation analysis")
    parser.add_argument("--seed-sensitivity", nargs="*", type=int, default=None,
                        help="Run across seeds, e.g. --seed-sensitivity 1 2 3 4 5 6")
    args = parser.parse_args()

    run_benchmark(
        out_dir=args.out,
        seed_override=args.seed,
        include_continuous=not args.no_continuous,
        include_ablation=not args.no_ablation,
        device=args.device,
    )

    if args.seed_sensitivity is not None:
        seeds = args.seed_sensitivity if len(args.seed_sensitivity) > 0 else [1, 2, 3, 4, 5, 6]
        run_seed_sensitivity(seeds, base_out=args.out)
