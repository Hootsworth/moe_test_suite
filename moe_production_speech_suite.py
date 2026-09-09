"""
Master 5-Seed Real Speech Production Benchmark Suite (Adult vs Child vs Dysarthric)
===================================================================================
Tests the original physical speech-production hypothesis across:
  - adult_speech (LibriSpeech profile)
  - child_speech (MyST / CSLU Kids profile)
  - dysarthric_speech (TORGO / UASpeech profile)
with strictly disjoint open-vocabulary validation transcripts and DeepSeek-style Hybrid MoE.
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
from moe_framework.hybrid_moe import HybridSequenceMoE
from moe_framework.losses import (
    DomainConditionalLoadBalanceLoss,
    StandardLoadBalanceLoss,
)
from moe_framework.metrics import (
    compute_gate_cosine_similarity,
    evaluate_hypothesis,
)
from moe_framework.production_speech_pipeline import (
    EnglishPhoneticTokenizer,
    ProductionSpeechWorld,
    create_production_dataloaders,
)

DOMAINS = ["adult_speech", "child_speech", "dysarthric_speech"]

DEFAULT_CONFIG = {
    "seed": 7,
    "in_dim": 80,
    "acoustic_dim": 4,
    "n_routed_experts": 4,
    "n_shared_experts": 1,  # DeepSeek-style shared invariant expert
    "top_k": 2,
    "hidden_dim": 48,
    "activation": "relu",
    "dropout": 0.0,
    "n_train_per_domain": 350,
    "n_val_per_domain": 80,
    "epochs": 35,
    "batch_size": 25,
    "lr": 0.007,
    "lambda_balance": 0.015,
}


def train_epoch(
    model: HybridSequenceMoE,
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
def evaluate_model(
    model: HybridSequenceMoE,
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    tokenizer: EnglishPhoneticTokenizer,
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


def train_single_run(
    gate_type: str,
    cfg: Dict[str, Union[int, float, str]],
    tokenizer: EnglishPhoneticTokenizer,
    train_loaders: Dict[str, torch.utils.data.DataLoader],
    val_loaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
) -> Tuple[HybridSequenceMoE, Dict[str, Dict[str, float]], Dict[str, np.ndarray], Dict[str, List[Tuple[str, str]]]]:
    model = HybridSequenceMoE(
        in_dim=int(cfg["in_dim"]),
        n_vocab=tokenizer.vocab_size - 1,
        n_routed_experts=int(cfg["n_routed_experts"]),
        n_shared_experts=int(cfg.get("n_shared_experts", 1)),
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
        train_epoch(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            ctc_loss_fn=ctc_loss_fn,
            bal_loss_fn=bal_loss_fn,
            device=device,
        )
        va_losses, va_metrics, gate_vecs, transcripts = evaluate_model(
            model=model,
            val_loaders=val_loaders,
            tokenizer=tokenizer,
            ctc_loss_fn=ctc_loss_fn,
            device=device,
        )
        final_metrics = va_metrics
        final_gate_vecs = gate_vecs
        final_transcripts = transcripts

    return model, final_metrics, final_gate_vecs, final_transcripts


def run_benchmark_single_seed(
    seed: int,
    cfg: Dict[str, Union[int, float]],
    device: str = "cpu",
    quiet: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, List[Tuple[str, str]]]]]:
    cfg = dict(cfg)
    cfg["seed"] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    world = ProductionSpeechWorld(cfg)
    tokenizer = world.tokenizer
    train_loaders, val_loaders = create_production_dataloaders(
        world=world,
        domains=DOMAINS,
        cfg=cfg,
        batch_size=int(cfg["batch_size"]),
    )

    gate_models = ["multi_gate", "single_gate", "continuous_acoustic"]
    summary_rows = []
    decomp_rows = []
    similarity_records = []
    transcripts_by_model = {}

    for gtype in gate_models:
        if not quiet:
            print(f"--> [Seed {seed}] Training '{gtype}'...")
        model, metrics, gate_vecs, transcripts = train_single_run(
            gate_type=gtype,
            cfg=cfg,
            tokenizer=tokenizer,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            device=device,
        )
        transcripts_by_model[gtype] = transcripts

        sim_rows = compute_gate_cosine_similarity(gate_vecs, model_gate_type=gtype)
        for r in sim_rows:
            r["seed"] = seed
        similarity_records.extend(sim_rows)

        for domain in DOMAINS:
            m = metrics[domain]
            summary_rows.append({
                "seed": seed,
                "model_gate_type": gtype,
                "domain": domain,
                "WER (%)": m["WER"],
                "CER (%)": m["CER"],
                "word_sub_rate (%)": m["word_sub_rate"],
                "word_del_rate (%)": m["word_del_rate"],
                "word_ins_rate (%)": m["word_ins_rate"],
            })

            decomp_rows.append({
                "seed": seed,
                "model_gate_type": gtype,
                "domain": domain,
                "substitutions": m["word_sub_rate"],
                "deletions": m["word_del_rate"],
                "insertions": m["word_ins_rate"],
                "total_wer": m["WER"],
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(decomp_rows), pd.DataFrame(similarity_records), transcripts_by_model


def run_full_5seed_production_benchmark(
    out_dir: str = "results_production_speech",
    seeds: Optional[List[int]] = None,
    device: Optional[str] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    seeds = seeds or [1, 2, 3, 4, 5]
    cfg = dict(DEFAULT_CONFIG)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_summaries = []
    all_decomps = []
    all_sims = []
    sample_transcripts_last_seed = None

    print(f"\n==========================================================================")
    print(f"   STARTING 5-SEED REAL SPEECH PRODUCTION BENCHMARK (SEEDS {seeds})       ")
    print(f"==========================================================================")

    t0 = time.time()
    for s in seeds:
        print(f"\n--- EXECUTING SEED {s} ---")
        df_sum, df_dec, df_sim, transcripts = run_benchmark_single_seed(
            seed=s, cfg=cfg, device=device, quiet=False
        )
        all_summaries.append(df_sum)
        all_decomps.append(df_dec)
        all_sims.append(df_sim)
        sample_transcripts_last_seed = transcripts

    elapsed = time.time() - t0

    df_full_summary = pd.concat(all_summaries, ignore_index=True)
    df_full_decomp = pd.concat(all_decomps, ignore_index=True)
    df_full_sim = pd.concat(all_sims, ignore_index=True)

    df_full_summary.to_csv(os.path.join(out_dir, "wer_cer_summary_across_seeds.csv"), index=False)
    df_full_decomp.to_csv(os.path.join(out_dir, "error_decomposition_across_seeds.csv"), index=False)
    df_full_sim.to_csv(os.path.join(out_dir, "gate_similarity_across_seeds.csv"), index=False)

    # 1. Print Raw Full Per-Seed Table
    print("\n===================================================================================================")
    print("                      FULL PER-SEED WER (%) & CER (%) BREAKDOWN (UNSEEN VAL SET)                   ")
    print("===================================================================================================")
    piv = df_full_summary.pivot(index=["domain", "seed"], columns="model_gate_type", values=["WER (%)", "CER (%)"])
    print(piv.round(2).to_string())

    # 2. Print Mean ± Std Aggregate Table
    print("\n===================================================================================================")
    print("                       5-SEED AGGREGATE SUMMARY TABLE (MEAN ± STD)                                 ")
    print("===================================================================================================")
    agg = df_full_summary.groupby(["model_gate_type", "domain"])[["WER (%)", "CER (%)", "word_sub_rate (%)", "word_del_rate (%)", "word_ins_rate (%)"]].agg(["mean", "std"]).round(2)
    print(agg.to_string())

    # 3. Direct Physical Hypothesis Verification Across Seeds
    print("\n===================================================================================================")
    print("      PHYSICAL HYPOTHESIS TEST: CHILD <-> DYSARTHRIC SIMILARITY vs. ADULT SIMILARITY               ")
    print("===================================================================================================")
    hyp_records = []
    for s in seeds:
        sub_sim = df_full_sim[df_full_sim["seed"] == s]
        for gtype in ["multi_gate", "continuous_acoustic"]:
            m_sub = sub_sim[sub_sim["model_gate_type"] == gtype]
            sim_map = {}
            for _, r in m_sub.iterrows():
                sim_map[(r["task_a"], r["task_b"])] = r["cosine_similarity"]
                sim_map[(r["task_b"], r["task_a"])] = r["cosine_similarity"]

            sim_ch_dys = sim_map.get(("child_speech", "dysarthric_speech"), 0.0)
            sim_ch_ad = sim_map.get(("child_speech", "adult_speech"), 0.0)
            sim_dys_ad = sim_map.get(("dysarthric_speech", "adult_speech"), 0.0)
            avg_ad = (sim_ch_ad + sim_dys_ad) / 2.0
            margin = sim_ch_dys - avg_ad
            hyp_records.append({
                "seed": s,
                "model_gate_type": gtype,
                "sim_child_dysarthric": round(float(sim_ch_dys), 4),
                "avg_sim_to_adult": round(float(avg_ad), 4),
                "margin": round(float(margin), 4),
                "hypothesis_supported": bool(sim_ch_dys > avg_ad),
            })

    df_hyp = pd.DataFrame(hyp_records)
    df_hyp.to_csv(os.path.join(out_dir, "hypothesis_check_across_seeds.csv"), index=False)
    print(df_hyp.to_string(index=False))

    for gtype in ["multi_gate", "continuous_acoustic"]:
        sub = df_hyp[df_hyp["model_gate_type"] == gtype]
        hit_rate = sub["hypothesis_supported"].sum()
        print(f"\n[{gtype}] Hypothesis Supported in {hit_rate}/{len(sub)} seeds ({hit_rate/len(sub)*100:.1f}%) | Mean Margin: {sub['margin'].mean():+.4f}")

    # 4. Print Sample Transcripts on Unseen Sentences
    print("\n===================================================================================================")
    print("      SAMPLE DECODED TRANSCRIPTS ON STRICTLY UNSEEN VALIDATION SENTENCES (SEED 5)                  ")
    print("===================================================================================================")
    for domain in DOMAINS:
        print(f"\n--- DOMAIN: {domain.upper()} ---")
        for gtype in ["multi_gate", "single_gate", "continuous_acoustic"]:
            print(f"  [{gtype}]")
            for idx, (ref, hyp) in enumerate(sample_transcripts_last_seed[gtype][domain][:3]):
                ed = levenshtein_distance(ref.split(), hyp.split())
                print(f"    Sample {idx+1} | Ref: '{ref}' | Hyp: '{hyp}' | S={ed['substitutions']}, D={ed['deletions']}, I={ed['insertions']}")

    print(f"\nBenchmark completed in {elapsed:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5-Seed Speech Production Benchmark")
    parser.add_argument("--out", default="results_production_speech", help="Output directory")
    parser.add_argument("--device", default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5], help="Random seeds")
    args = parser.parse_args()

    run_full_5seed_production_benchmark(
        out_dir=args.out,
        seeds=args.seeds,
        device=args.device,
    )
