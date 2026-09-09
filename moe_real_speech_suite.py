"""
PyTorch Real-World Speech Multi-Gate MoE Benchmark Suite (Project Vaani)
========================================================================
Comprehensive real-world multi-domain speech benchmark evaluating:
  - Project Vaani domains: Standard Hindi, Bhojpuri Regional Dialect, Rural/Atypical Field
  - 80-dimensional Log-Mel Filterbank acoustic features
  - Full Devanagari CTC vocabulary decoding
  - Word Error Rate (WER), Character Error Rate (CER), and S/D/I error decomposition
  - Cross-domain gate similarity & expert ablation analysis
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
from moe_framework.real_data_pipeline import (
    DevanagariTokenizer,
    VaaniSpeechWorld,
    create_vaani_dataloaders,
)
from moe_framework.sequence_models import SequenceMultiGateMoE

DOMAINS = ["standard_hindi", "bhojpuri_dialect", "rural_field_atypical"]

DEFAULT_REAL_CONFIG = {
    "seed": 7,
    "in_dim": 80,  # 80-dim log-mel filterbanks
    "acoustic_dim": 4,
    "n_routed_experts": 5,
    "n_shared_experts": 0,
    "top_k": 2,
    "hidden_dim": 48,
    "activation": "relu",
    "dropout": 0.0,
    "n_train_real_per_domain": 300,
    "n_val_real_per_domain": 75,
    "epochs": 30,
    "batch_size": 25,
    "lr": 0.006,
    "lambda_balance": 0.015,
}


def train_real_epoch(
    model: SequenceMultiGateMoE,
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.Optimizer,
    ctc_loss_fn: nn.Module,
    bal_loss_fn: Optional[nn.Module],
    device: str,
) -> Dict[str, float]:
    model.train()
    task_losses = {d: [] for d in DOMAINS}

    iterators = {d: iter(train_loaders[d]) for d in DOMAINS}
    min_batches = min(len(loader) for loader in train_loaders.values())

    for _ in range(min_batches):
        optimizer.zero_grad()
        task_G_dict, task_mask_dict = {}, {}
        total_ctc_loss = torch.tensor(0.0, device=device)

        for domain in DOMAINS:
            batch = next(iterators[domain])
            x = batch["x"].to(device)
            targets = batch["targets"].to(device)
            in_lens = batch["input_lengths"].to(device)
            tar_lens = batch["target_lengths"].to(device)
            z = batch["z_acoustic"].to(device)

            out = model(x=x, input_lengths=in_lens, task=domain, acoustic_emb=z)
            log_probs = out["log_probs_ctc"]
            sub_lengths = out["sub_lengths"]

            loss = ctc_loss_fn(log_probs, targets, sub_lengths, tar_lens)
            total_ctc_loss = total_ctc_loss + loss
            task_losses[domain].append(loss.item())

            task_G_dict[domain] = out["G"]
            task_mask_dict[domain] = out["mask"]

        bal_loss = torch.tensor(0.0, device=device)
        if bal_loss_fn is not None:
            bal_loss, _ = bal_loss_fn(task_G_dict, task_mask_dict, model.n_routed_experts)

        total_loss = total_ctc_loss + bal_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    return {d: float(np.mean(task_losses[d])) for d in DOMAINS}


@torch.no_grad()
def evaluate_real_model(
    model: SequenceMultiGateMoE,
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    tokenizer: DevanagariTokenizer,
    ctc_loss_fn: nn.Module,
    device: str,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, List[Tuple[str, str]]]]:
    model.eval()
    val_losses = {}
    val_metrics = {}
    gate_vectors = {}
    sample_transcripts = {}

    for domain in DOMAINS:
        losses = []
        evaluator = SequenceErrorEvaluator(vocab_map=tokenizer.id_to_char, space_token=tokenizer.space_id)
        domain_Gm_list = []
        domain_samples = []

        for batch in val_loaders[domain]:
            x = batch["x"].to(device)
            targets = batch["targets"].to(device)
            in_lens = batch["input_lengths"].to(device)
            tar_lens = batch["target_lengths"].to(device)
            z = batch["z_acoustic"].to(device)

            out = model(x=x, input_lengths=in_lens, task=domain, acoustic_emb=z)
            sub_lens = out["sub_lengths"]
            loss = ctc_loss_fn(out["log_probs_ctc"], targets, sub_lens, tar_lens)
            losses.append(loss.item())

            domain_Gm_list.append(out["Gm"].cpu().numpy())

            hyp_tokens = ctc_greedy_decode(out["logits"], sub_lens, blank_idx=0)

            ref_tokens = []
            for b in range(targets.size(0)):
                u = int(tar_lens[b].item())
                ref_seq = targets[b, :u].cpu().tolist()
                ref_tokens.append(ref_seq)
                if len(domain_samples) < 4:
                    ref_text = tokenizer.decode(ref_seq)
                    hyp_text = tokenizer.decode(hyp_tokens[b])
                    domain_samples.append((ref_text, hyp_text))

            evaluator.update(ref_tokens, hyp_tokens)

        val_losses[domain] = float(np.mean(losses)) if losses else 0.0
        val_metrics[domain] = evaluator.compute()
        sample_transcripts[domain] = domain_samples

        all_Gm = np.concatenate(domain_Gm_list, axis=0)
        gate_vectors[domain] = all_Gm.mean(axis=0)

    return val_losses, val_metrics, gate_vectors, sample_transcripts


def train_single_real_model(
    gate_type: str,
    cfg: Dict[str, Union[int, float, str]],
    tokenizer: DevanagariTokenizer,
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
    curve_log: List[Dict[str, Union[str, int, float]]],
) -> Tuple[SequenceMultiGateMoE, Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, List[Tuple[str, str]]]]:
    model = SequenceMultiGateMoE(
        in_dim=int(cfg["in_dim"]),
        n_vocab=tokenizer.vocab_size - 1,
        n_routed_experts=int(cfg["n_routed_experts"]),
        n_shared_experts=int(cfg.get("n_shared_experts", 0)),
        top_k=int(cfg["top_k"]),
        hidden_dim=int(cfg["hidden_dim"]),
        gate_type=gate_type,
        tasks=DOMAINS,
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
        tr_losses = train_real_epoch(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            ctc_loss_fn=ctc_loss_fn,
            bal_loss_fn=bal_loss_fn,
            device=device,
        )
        va_losses, va_metrics, gate_vecs, transcripts = evaluate_real_model(
            model=model,
            val_loaders=val_loaders,
            tokenizer=tokenizer,
            ctc_loss_fn=ctc_loss_fn,
            device=device,
        )

        final_metrics = va_metrics
        final_gate_vecs = gate_vecs
        final_transcripts = transcripts

        for domain in DOMAINS:
            curve_log.append({
                "model_gate_type": gate_type,
                "domain": domain,
                "epoch": epoch,
                "train_ctc_loss": round(tr_losses[domain], 4),
                "val_ctc_loss": round(va_losses[domain], 4),
                "val_WER": va_metrics[domain]["WER"],
                "val_CER": va_metrics[domain]["CER"],
            })

    return model, final_metrics, final_gate_vecs, final_transcripts


def run_vaani_benchmark(
    out_dir: str = "results_real_speech",
    seed_override: Optional[int] = None,
    include_continuous: bool = True,
    device: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(DEFAULT_REAL_CONFIG)
    if seed_override is not None:
        cfg["seed"] = seed_override

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    world = VaaniSpeechWorld(cfg)
    tokenizer = world.tokenizer
    train_loaders, val_loaders = create_vaani_dataloaders(
        world=world,
        domains=DOMAINS,
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
            print(f"--> Training Real-Speech MoE on Project Vaani with gate_type: '{gtype}' on {device}...")
        model, metrics, gate_vecs, transcripts = train_single_real_model(
            gate_type=gtype,
            cfg=cfg,
            tokenizer=tokenizer,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            device=device,
            curve_log=curve_log,
        )
        all_transcripts[gtype] = transcripts

        sim_rows = compute_gate_cosine_similarity(gate_vecs, model_gate_type=gtype)
        similarity_records.extend(sim_rows)

        for domain in DOMAINS:
            m = metrics[domain]
            summary_rows.append({
                "model_gate_type": gtype,
                "domain": domain,
                "WER (%)": m["WER"],
                "CER (%)": m["CER"],
                "word_sub_rate (%)": m["word_sub_rate"],
                "word_del_rate (%)": m["word_del_rate"],
                "word_ins_rate (%)": m["word_ins_rate"],
            })

            decomp_rows.append({
                "model_gate_type": gtype,
                "domain": domain,
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

    df_curves.to_csv(os.path.join(out_dir, "real_training_curves.csv"), index=False)
    df_summary.to_csv(os.path.join(out_dir, "real_wer_cer_summary.csv"), index=False)
    df_decomp.to_csv(os.path.join(out_dir, "real_error_decomposition.csv"), index=False)
    df_sim.to_csv(os.path.join(out_dir, "real_gate_similarity.csv"), index=False)

    if not quiet:
        print(f"\nReal-Speech Vaani Training completed in {elapsed:.2f}s on {device}.")
        print("\n=======================================================")
        print("    SAMPLE DEVANAGARI TRANSCRIPTS (REF vs HYP)         ")
        print("=======================================================")
        for domain in DOMAINS:
            print(f"\n--- DOMAIN: {domain.upper()} ---")
            for gtype in gate_models:
                print(f"  [{gtype}]")
                samples = all_transcripts[gtype][domain]
                for idx, (ref, hyp) in enumerate(samples[:3]):
                    print(f"    Sample {idx+1} | Ref: '{ref}' | Hyp: '{hyp}'")

        print("\n=== Word Error Rate (WER) & Character Error Rate (CER) on Project Vaani ===")
        print(df_summary.to_string(index=False))
        print("\n=== Error Decomposition (%) on Real Audio Features ===")
        print(df_decomp.to_string(index=False))
        print("\n=== Real-Speech Gate Cosine Similarity Matrix ===")
        print(df_sim.to_string(index=False))

    return {
        "summary": df_summary,
        "decomposition": df_decomp,
        "similarity": df_sim,
    }


def run_vaani_seed_sensitivity(
    seeds: List[int],
    base_out: str = "results_real_speech",
) -> pd.DataFrame:
    rows = []
    seed_dir = os.path.join(base_out, "seed_sensitivity")
    os.makedirs(seed_dir, exist_ok=True)

    print(f"\n[Multi-Seed Vaani Sensitivity Sweep] Testing seeds: {seeds}...")
    for s in seeds:
        print(f"--> Running Vaani seed={s} ...")
        res = run_vaani_benchmark(
            out_dir=os.path.join(seed_dir, f"seed_{s}"),
            seed_override=s,
            include_continuous=True,
            quiet=True,
        )
        sim_df = res["similarity"]
        m_sim = sim_df[sim_df["model_gate_type"] == "multi_gate"]
        for _, r in m_sim.iterrows():
            row = dict(r)
            row["seed"] = s
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base_out, "gate_similarity_across_seeds.csv"), index=False)
    print("\n=== Multi-Seed Gate Similarity Summary on Real Speech ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-World Speech MoE Benchmark on Project Vaani")
    parser.add_argument("--out", default="results_real_speech", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--no-continuous", action="store_true", help="Skip continuous acoustic gate")
    parser.add_argument("--seed-sensitivity", nargs="*", type=int, default=None,
                        help="Run across seeds, e.g. --seed-sensitivity 1 2 3 4 5 6")
    args = parser.parse_args()

    run_vaani_benchmark(
        out_dir=args.out,
        seed_override=args.seed,
        include_continuous=not args.no_continuous,
        device=args.device,
    )

    if args.seed_sensitivity is not None:
        seeds = args.seed_sensitivity if len(args.seed_sensitivity) > 0 else [1, 2, 3, 4, 5, 6]
        run_vaani_seed_sensitivity(seeds, base_out=args.out)
