"""
PyTorch Sequence-Level Multi-Gate MoE ASR Benchmark Suite
=========================================================
End-to-end training and evaluation of Multi-Gate MoE on continuous speech
utterances using Connectionist Temporal Classification (CTC) loss with
2x Conv1D subsampling frontend, evaluating Word Error Rate (WER),
Character Error Rate (CER), and S/D/I error decomposition.
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from moe_framework.ctc_engine import (
    SequenceErrorEvaluator,
    ctc_greedy_decode,
    levenshtein_distance,
)
from moe_framework.losses import (
    DomainConditionalLoadBalanceLoss,
    StandardLoadBalanceLoss,
)
from moe_framework.metrics import (
    compute_gate_cosine_similarity,
    evaluate_hypothesis,
)
from moe_framework.sequence_data import (
    SpeechSequenceWorld,
    create_sequence_dataloaders,
)
from moe_framework.sequence_models import SequenceMultiGateMoE

TASKS = ["language", "child", "atypical"]

DEFAULT_SEQ_CONFIG = {
    "seed": 7,
    "n_vocab": 12,
    "dim_a": 10,
    "dim_b": 10,
    "acoustic_dim": 4,
    "n_routed_experts": 5,
    "n_shared_experts": 0,
    "top_k": 2,
    "hidden_dim": 32,
    "activation": "relu",
    "dropout": 0.0,
    "n_train_seq_per_task": 400,
    "n_val_seq_per_task": 100,
    "epochs": 35,
    "batch_size": 25,
    "lr": 0.008,
    "lambda_balance": 0.01,
    "noise_std": 0.7,
    "class_scale": 1.6,
    "alpha_mild": 0.15,
    "alpha_strong": 0.8,
    "mag_child": 1.0,
    "mag_atypical": 1.25,
    "mag_language": 0.0,
}


def train_seq_epoch(
    model: SequenceMultiGateMoE,
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    ctc_loss_fn: nn.Module,
    bal_loss_fn: Optional[nn.Module],
    cfg: Dict[str, Union[int, float, str]],
    device: str,
) -> Dict[str, float]:
    model.train()
    task_losses = {t: [] for t in TASKS}

    iterators = {t: iter(train_loaders[t]) for t in TASKS}
    min_batches = min(len(loader) for loader in train_loaders.values())

    for _ in range(min_batches):
        optimizer.zero_grad()
        task_G_dict, task_mask_dict = {}, {}
        total_ctc_loss = torch.tensor(0.0, device=device)

        for task in TASKS:
            batch = next(iterators[task])
            x = batch["x"].to(device)
            targets = batch["targets"].to(device)
            in_lens = batch["input_lengths"].to(device)
            tar_lens = batch["target_lengths"].to(device)
            z = batch["z_acoustic"].to(device)

            out = model(x=x, input_lengths=in_lens, task=task, acoustic_emb=z)
            log_probs = out["log_probs_ctc"]  # (T_sub, B, V+1)
            sub_lengths = out["sub_lengths"]  # (B,)

            loss = ctc_loss_fn(log_probs, targets, sub_lengths, tar_lens)
            total_ctc_loss = total_ctc_loss + loss
            task_losses[task].append(loss.item())

            task_G_dict[task] = out["G"]
            task_mask_dict[task] = out["mask"]

        # Domain conditional load balancing
        bal_loss = torch.tensor(0.0, device=device)
        if bal_loss_fn is not None:
            bal_loss, _ = bal_loss_fn(task_G_dict, task_mask_dict, model.n_routed_experts)

        total_loss = total_ctc_loss + bal_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    return {t: float(np.mean(task_losses[t])) for t in TASKS}


@torch.no_grad()
def evaluate_seq_model(
    model: SequenceMultiGateMoE,
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    ctc_loss_fn: nn.Module,
    device: str,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, List[Tuple[List[int], List[int]]]]]:
    model.eval()
    val_losses = {}
    val_metrics = {}
    gate_vectors = {}
    sample_transcripts = {}

    for task in TASKS:
        losses = []
        evaluator = SequenceErrorEvaluator()
        task_Gm_list = []
        task_samples = []

        for batch in val_loaders[task]:
            x = batch["x"].to(device)
            targets = batch["targets"].to(device)
            in_lens = batch["input_lengths"].to(device)
            tar_lens = batch["target_lengths"].to(device)
            z = batch["z_acoustic"].to(device)

            out = model(x=x, input_lengths=in_lens, task=task, acoustic_emb=z)
            sub_lens = out["sub_lengths"]
            loss = ctc_loss_fn(out["log_probs_ctc"], targets, sub_lens, tar_lens)
            losses.append(loss.item())

            task_Gm_list.append(out["Gm"].cpu().numpy())

            # Decode predictions
            hyp_tokens = ctc_greedy_decode(out["logits"], sub_lens, blank_idx=0)

            # Extract ground-truth reference token lists
            ref_tokens = []
            for b in range(targets.size(0)):
                u = int(tar_lens[b].item())
                ref_seq = targets[b, :u].cpu().tolist()
                ref_tokens.append(ref_seq)
                if len(task_samples) < 5:
                    task_samples.append((ref_seq, hyp_tokens[b]))

            evaluator.update(ref_tokens, hyp_tokens)

        val_losses[task] = float(np.mean(losses)) if losses else 0.0
        val_metrics[task] = evaluator.compute()
        sample_transcripts[task] = task_samples

        all_Gm = np.concatenate(task_Gm_list, axis=0)
        gate_vectors[task] = all_Gm.mean(axis=0)

    return val_losses, val_metrics, gate_vectors, sample_transcripts


def train_single_seq_model(
    gate_type: str,
    cfg: Dict[str, Union[int, float, str]],
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
    curve_log: List[Dict[str, Union[str, int, float]]],
) -> Tuple[SequenceMultiGateMoE, Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, List[Tuple[List[int], List[int]]]]]:
    in_dim = int(cfg["dim_a"]) + int(cfg["dim_b"])
    model = SequenceMultiGateMoE(
        in_dim=in_dim,
        n_vocab=int(cfg["n_vocab"]),
        n_routed_experts=int(cfg["n_routed_experts"]),
        n_shared_experts=int(cfg.get("n_shared_experts", 0)),
        top_k=int(cfg["top_k"]),
        hidden_dim=int(cfg["hidden_dim"]),
        gate_type=gate_type,
        tasks=TASKS,
        acoustic_dim=int(cfg["acoustic_dim"]),
        activation=str(cfg.get("activation", "relu")),
        dropout=float(cfg.get("dropout", 0.0)),
        init_blank_bias=-2.0,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    bal_loss_fn = DomainConditionalLoadBalanceLoss(lambda_balance=float(cfg["lambda_balance"]))

    final_metrics = {}
    final_gate_vecs = {}
    final_transcripts = {}

    for epoch in range(int(cfg["epochs"])):
        tr_losses = train_seq_epoch(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            ctc_loss_fn=ctc_loss_fn,
            bal_loss_fn=bal_loss_fn,
            cfg=cfg,
            device=device,
        )
        va_losses, va_metrics, gate_vecs, transcripts = evaluate_seq_model(
            model=model,
            val_loaders=val_loaders,
            ctc_loss_fn=ctc_loss_fn,
            device=device,
        )

        final_metrics = va_metrics
        final_gate_vecs = gate_vecs
        final_transcripts = transcripts

        for task in TASKS:
            curve_log.append({
                "model_gate_type": gate_type,
                "task": task,
                "epoch": epoch,
                "train_ctc_loss": round(tr_losses[task], 4),
                "val_ctc_loss": round(va_losses[task], 4),
                "val_WER": va_metrics[task]["WER"],
                "val_CER": va_metrics[task]["CER"],
            })

    return model, final_metrics, final_gate_vecs, final_transcripts


def run_sequence_benchmark(
    out_dir: str = "results_seq_wer",
    seed_override: Optional[int] = None,
    include_continuous: bool = True,
    device: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(DEFAULT_SEQ_CONFIG)
    if seed_override is not None:
        cfg["seed"] = seed_override

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    world = SpeechSequenceWorld(cfg)
    train_loaders, val_loaders = create_sequence_dataloaders(
        world=world,
        tasks=TASKS,
        cfg=cfg,
        batch_size=int(cfg["batch_size"]),
    )

    gate_models = ["multi_gate", "single_gate"]
    if include_continuous:
        gate_models.append("continuous_acoustic")

    curve_log = []
    summary_rows = []
    decomp_rows = []
    similarity_records = []
    all_transcripts = {}

    t0 = time.time()
    for gtype in gate_models:
        if not quiet:
            print(f"--> Training Sequence MoE with gate_type: '{gtype}' on {device}...")
        model, metrics, gate_vecs, transcripts = train_single_seq_model(
            gate_type=gtype,
            cfg=cfg,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            device=device,
            curve_log=curve_log,
        )
        all_transcripts[gtype] = transcripts

        sim_rows = compute_gate_cosine_similarity(gate_vecs, model_gate_type=gtype)
        similarity_records.extend(sim_rows)

        for task in TASKS:
            m = metrics[task]
            summary_rows.append({
                "model_gate_type": gtype,
                "task": task,
                "WER (%)": m["WER"],
                "CER (%)": m["CER"],
                "word_sub_rate (%)": m["word_sub_rate"],
                "word_del_rate (%)": m["word_del_rate"],
                "word_ins_rate (%)": m["word_ins_rate"],
            })

            decomp_rows.append({
                "model_gate_type": gtype,
                "task": task,
                "substitutions": m["word_sub_rate"],
                "deletions": m["word_del_rate"],
                "insertions": m["word_ins_rate"],
                "total_wer": m["WER"],
            })

    elapsed = time.time() - t0

    # DataFrames
    df_curves = pd.DataFrame(curve_log)
    df_summary = pd.DataFrame(summary_rows)
    df_decomp = pd.DataFrame(decomp_rows)
    df_sim = pd.DataFrame(similarity_records)

    df_curves.to_csv(os.path.join(out_dir, "sequence_training_curves.csv"), index=False)
    df_summary.to_csv(os.path.join(out_dir, "wer_cer_summary.csv"), index=False)
    df_decomp.to_csv(os.path.join(out_dir, "error_decomposition.csv"), index=False)
    df_sim.to_csv(os.path.join(out_dir, "gate_similarity_seq.csv"), index=False)

    # Hypothesis check
    hyp_rows = []
    for gtype in ["multi_gate", "continuous_acoustic"]:
        sub_sim = [r for r in similarity_records if r["model_gate_type"] == gtype]
        if sub_sim:
            hyp_res = evaluate_hypothesis(sub_sim)
            hyp_res["model_gate_type"] = gtype
            hyp_rows.append(hyp_res)
    df_hyp = pd.DataFrame(hyp_rows)
    df_hyp.to_csv(os.path.join(out_dir, "hypothesis_check_seq.csv"), index=False)

    if not quiet:
        print(f"\nSequence Training completed in {elapsed:.2f}s on {device}.")
        print("\n=======================================================")
        print("          SAMPLE REFERENCE VS HYPOTHESIS TRANSCRIPTS   ")
        print("=======================================================")
        for task in TASKS:
            print(f"\n--- TASK: {task.upper()} ---")
            for gtype in gate_models:
                print(f"  [{gtype}]")
                samples = all_transcripts[gtype][task]
                for idx, (ref, hyp) in enumerate(samples[:4]):
                    ed = levenshtein_distance(ref, hyp)
                    print(f"    Sample {idx+1} | Ref: {ref} | Hyp: {hyp} | S={ed['substitutions']}, D={ed['deletions']}, I={ed['insertions']}")

        print("\n=== Word Error Rate (WER) & Character Error Rate (CER) Summary ===")
        print(df_summary.to_string(index=False))
        print("\n=== Error Decomposition: Substitutions / Deletions / Insertions (%) ===")
        print(df_decomp.to_string(index=False))
        print("\n=== Sequence Gate Cosine Similarity Matrix ===")
        print(df_sim.to_string(index=False))
        print("\n=== Sequence Hypothesis Check ===")
        print(df_hyp.to_string(index=False))

    return {
        "summary": df_summary,
        "decomposition": df_decomp,
        "similarity": df_sim,
        "hypothesis": df_hyp,
    }


def run_sequence_seed_sensitivity(
    seeds: List[int],
    base_out: str = "results_seq_wer",
) -> pd.DataFrame:
    rows = []
    seed_dir = os.path.join(base_out, "seed_sensitivity")
    os.makedirs(seed_dir, exist_ok=True)

    print(f"\n[Multi-Seed Sequence Sensitivity Sweep] Testing seeds: {seeds}...")
    for s in seeds:
        print(f"--> Running sequence seed={s} ...")
        res = run_sequence_benchmark(
            out_dir=os.path.join(seed_dir, f"seed_{s}"),
            seed_override=s,
            include_continuous=True,
            quiet=True,
        )
        for _, hyp_row in res["hypothesis"].iterrows():
            record = dict(hyp_row)
            record["seed"] = s
            rows.append(record)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base_out, "hypothesis_across_seeds_seq.csv"), index=False)

    print("\n=== Multi-Seed Sequence Hypothesis Results ===")
    print(df[["seed", "model_gate_type", "similarity_child_atypical", "avg_similarity_to_language", "margin", "hypothesis_supported"]].to_string(index=False))

    for gtype in ["multi_gate", "continuous_acoustic"]:
        sub = df[df["model_gate_type"] == gtype]
        if not sub.empty:
            hit_rate = sub["hypothesis_supported"].sum()
            print(f"[{gtype}] Hypothesis Supported: {hit_rate}/{len(sub)} ({hit_rate/len(sub)*100:.1f}%) | Mean Margin: {sub['margin'].mean():.4f}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence-Level Multi-Gate MoE CTC Benchmark")
    parser.add_argument("--out", default="results_seq_wer", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--no-continuous", action="store_true", help="Skip continuous acoustic gate")
    parser.add_argument("--seed-sensitivity", nargs="*", type=int, default=None,
                        help="Run across seeds, e.g. --seed-sensitivity 1 2 3 4 5 6")
    args = parser.parse_args()

    run_sequence_benchmark(
        out_dir=args.out,
        seed_override=args.seed,
        include_continuous=not args.no_continuous,
        device=args.device,
    )

    if args.seed_sensitivity is not None:
        seeds = args.seed_sensitivity if len(args.seed_sensitivity) > 0 else [1, 2, 3, 4, 5, 6]
        run_sequence_seed_sensitivity(seeds, base_out=args.out)
