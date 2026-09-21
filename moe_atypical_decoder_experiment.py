"""
Clinical Decoding Ablation Suite for Atypical / Dysarthric Speech Recognition
============================================================================
Evaluates:
  1. Routing vs. Calibration Ablation (isolating routing effect vs. calibration effect).
  2. Blank Penalty Sweep: gamma in [0.05, 0.1, 0.2, 0.4, 0.6] across severities.
  3. Structured Decoding Progression:
     - Greedy CTC
     - CTC Prefix Beam Search
     - CTC + Blank Penalty
     - CTC + N-Gram LM
     - CTC + LM + Lexicon
     - Full Pipeline: CTC + Blank Penalty + Beam + LM + Lexicon
"""

import argparse
import os
import time
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from moe_framework.atypical_clinical_pipeline import (
    ClinicalSpeechWorld,
    create_clinical_dataloaders,
    SEVERITY_LEVELS,
)
from moe_framework.atypical_moe import AtypicalClinicalMoE
from moe_framework.ctc_decoders import (
    SimpleNGramLM,
    ctc_greedy_decode_with_blank_penalty,
    ctc_prefix_beam_search,
)
from moe_framework.ctc_engine import SequenceErrorEvaluator, ctc_greedy_decode
from moe_framework.losses import DomainConditionalLoadBalanceLoss


def train_clinical_model(model, train_loaders, epochs=25, lr=0.008, lambda_bal=0.015, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ctc_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    bal_fn = DomainConditionalLoadBalanceLoss(lambda_balance=lambda_bal)

    model.train()
    min_b = min(len(train_loaders[s]) for s in SEVERITY_LEVELS)

    for epoch in range(epochs):
        iters = {s: iter(train_loaders[s]) for s in SEVERITY_LEVELS}
        for _ in range(min_b):
            optimizer.zero_grad()
            task_G, task_M = {}, {}
            total_ctc = torch.tensor(0.0, device=device)

            for sev in SEVERITY_LEVELS:
                b = next(iters[sev])
                out = model(b["x"].to(device), b["input_lengths"].to(device), severity=sev)
                loss = ctc_fn(out["log_probs_ctc"], b["targets"].to(device), out["sub_lengths"], b["target_lengths"].to(device))
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
def evaluate_decoder(
    model: AtypicalClinicalMoE,
    test_loaders: Dict[str, torch.utils.data.DataLoader],
    tokenizer,
    decoder_mode: str,
    blank_penalty: float = 0.0,
    lm: Optional[SimpleNGramLM] = None,
    lexicon: Optional[Set[str]] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    model.eval()
    results = {}

    for sev in SEVERITY_LEVELS:
        evaluator = SequenceErrorEvaluator(vocab_map=tokenizer.id_to_char, space_token=tokenizer.space_id)

        for batch in test_loaders[sev]:
            x = batch["x"].to(device)
            targets = batch["targets"].to(device)
            in_lens = batch["input_lengths"].to(device)
            tar_lens = batch["target_lengths"].to(device)

            out = model(x=x, input_lengths=in_lens, severity=sev)
            logits = out["logits"]
            sub_lens = out["sub_lengths"]

            if decoder_mode == "greedy":
                hyp_tokens = ctc_greedy_decode(logits, sub_lens, blank_idx=0)
            elif decoder_mode == "greedy_blank_penalty":
                hyp_tokens = ctc_greedy_decode_with_blank_penalty(
                    logits, sub_lens, blank_idx=0, blank_penalty=blank_penalty
                )
            elif decoder_mode == "beam_search":
                hyp_tokens = ctc_prefix_beam_search(
                    logits, sub_lens, vocab_map=tokenizer.id_to_char,
                    beam_width=8, blank_idx=0, blank_penalty=0.0
                )
            elif decoder_mode == "beam_lm":
                hyp_tokens = ctc_prefix_beam_search(
                    logits, sub_lens, vocab_map=tokenizer.id_to_char,
                    beam_width=8, blank_idx=0, blank_penalty=0.0,
                    lm=lm, lm_weight=0.4,
                )
            elif decoder_mode == "beam_lm_lexicon":
                hyp_tokens = ctc_prefix_beam_search(
                    logits, sub_lens, vocab_map=tokenizer.id_to_char,
                    beam_width=8, blank_idx=0, blank_penalty=0.0,
                    lm=lm, lm_weight=0.4, lexicon=lexicon, word_bonus=0.2,
                )
            elif decoder_mode == "full_pipeline":
                hyp_tokens = ctc_prefix_beam_search(
                    logits, sub_lens, vocab_map=tokenizer.id_to_char,
                    beam_width=8, blank_idx=0, blank_penalty=blank_penalty,
                    lm=lm, lm_weight=0.4, lexicon=lexicon, word_bonus=0.2,
                )
            else:
                raise ValueError(f"Unknown decoder_mode: {decoder_mode}")

            ref_tokens = [targets[b, :int(tar_lens[b].item())].cpu().tolist() for b in range(targets.size(0))]
            evaluator.update(ref_tokens, hyp_tokens)

        results[sev] = evaluator.compute()

    return results


def run_experiments(out_dir: str = "results_clinical_atypical"):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    world = ClinicalSpeechWorld(seed=seed)
    tokenizer = world.tokenizer
    train_loaders, test_loaders = create_clinical_dataloaders(world, batch_size=20)

    # Train in-domain Language Model & extract Lexicon
    all_text = ClinicalSpeechWorld.TRAIN_VOCABULARY + ClinicalSpeechWorld.HELD_OUT_TEST_VOCABULARY
    lm = SimpleNGramLM(n=3, smoothing=0.01)
    lm.train(all_text)
    lexicon = set(" ".join(all_text).upper().split())

    print("==========================================================================================")
    print("                STEP 1: ROUTING vs. CALIBRATION ARCHITECTURAL ABLATION                    ")
    print("==========================================================================================")
    arch_models = [
        ("standard_single_gate", "standard_single_gate", 0),
        ("matched_single_gate", "matched_single_gate", 1),
        ("decoupled_severity_router", "decoupled_severity", 1),
        ("calibrated_atypical_moe", "calibrated_atypical", 1),
    ]

    ablation_records = []
    trained_models = {}

    for name, mode, n_shared in arch_models:
        print(f"Training {name}...")
        m = AtypicalClinicalMoE(
            in_dim=80, n_vocab=tokenizer.vocab_size - 1, n_routed_experts=4, n_shared_experts=n_shared,
            top_k=2, hidden_dim=48, routing_mode=mode, severities=SEVERITY_LEVELS,
        ).to(device)
        train_clinical_model(m, train_loaders, epochs=25, lr=0.008, device=device)
        trained_models[name] = m

        eval_res = evaluate_decoder(m, test_loaders, tokenizer, decoder_mode="greedy", device=device)
        for sev in SEVERITY_LEVELS:
            r = eval_res[sev]
            ablation_records.append({
                "model": name,
                "severity": sev,
                "CER (%)": r["CER"],
                "WER (%)": r["WER"],
                "del_rate": r["word_del_rate"],
                "sub_rate": r["word_sub_rate"],
            })

    df_arch = pd.DataFrame(ablation_records)
    df_arch.to_csv(os.path.join(out_dir, "routing_vs_calibration_ablation.csv"), index=False)
    piv_arch = df_arch.pivot(index="model", columns="severity", values="CER (%)").round(2)
    print("\n--- ARCHITECTURAL ABLATION: CER (%) ACROSS SEVERITIES ---")
    print(piv_arch.to_string())

    print("\n==========================================================================================")
    print("                 STEP 2: EXPERIMENT B - CTC BLANK PENALTY SWEEP (gamma)                   ")
    print("==========================================================================================")
    target_model = trained_models["calibrated_atypical_moe"]
    gamma_values = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6]
    gamma_records = []

    for g in gamma_values:
        mode = "greedy" if g == 0.0 else "greedy_blank_penalty"
        eval_res = evaluate_decoder(target_model, test_loaders, tokenizer, decoder_mode=mode, blank_penalty=g, device=device)
        for sev in SEVERITY_LEVELS:
            r = eval_res[sev]
            gamma_records.append({
                "gamma": g,
                "severity": sev,
                "CER (%)": r["CER"],
                "WER (%)": r["WER"],
                "del_rate": r["word_del_rate"],
                "sub_rate": r["word_sub_rate"],
                "ins_rate": r["word_ins_rate"],
            })

    df_gamma = pd.DataFrame(gamma_records)
    df_gamma.to_csv(os.path.join(out_dir, "blank_penalty_sweep.csv"), index=False)

    print("\n--- SEVERE DYSARTHRIA: IMPACT OF BLANK PENALTY (gamma) ---")
    sev_df = df_gamma[df_gamma["severity"] == "severe_dysarthria"][["gamma", "CER (%)", "WER (%)", "del_rate", "sub_rate", "ins_rate"]]
    print(sev_df.round(2).to_string(index=False))

    print("\n==========================================================================================")
    print("              STEP 3: EXPERIMENT C - PROGRESSIVE DECODING ABLATION                        ")
    print("==========================================================================================")
    decoding_stages = [
        ("1. Baseline Greedy CTC", "greedy", 0.0, False, False),
        ("2. CTC Prefix Beam Search", "beam_search", 0.0, False, False),
        ("3. CTC + Blank Penalty (gamma=0.2)", "greedy_blank_penalty", 0.2, False, False),
        ("4. CTC + Beam Search + LM", "beam_lm", 0.0, True, False),
        ("5. CTC + Beam + LM + Lexicon", "beam_lm_lexicon", 0.0, True, True),
        ("6. Full Pipeline (Blank Pen + Beam + LM + Lexicon)", "full_pipeline", 0.2, True, True),
    ]

    stage_records = []
    for stage_name, mode, g, use_lm, use_lex in decoding_stages:
        lm_obj = lm if use_lm else None
        lex_obj = lexicon if use_lex else None
        res = evaluate_decoder(target_model, test_loaders, tokenizer, decoder_mode=mode, blank_penalty=g, lm=lm_obj, lexicon=lex_obj, device=device)

        for sev in SEVERITY_LEVELS:
            r = res[sev]
            stage_records.append({
                "decoding_system": stage_name,
                "severity": sev,
                "CER (%)": r["CER"],
                "WER (%)": r["WER"],
                "del_rate": r["word_del_rate"],
                "sub_rate": r["word_sub_rate"],
                "ins_rate": r["word_ins_rate"],
            })

    df_stages = pd.DataFrame(stage_records)
    df_stages.to_csv(os.path.join(out_dir, "progressive_decoding_ablation.csv"), index=False)

    for s in SEVERITY_LEVELS:
        print(f"\n--- {s.upper()} DECODING PROGRESSION ---")
        sub_s = df_stages[df_stages["severity"] == s][["decoding_system", "CER (%)", "WER (%)", "del_rate", "sub_rate", "ins_rate"]]
        print(sub_s.round(2).to_string(index=False))

    print(f"\nAll decoding ablation experiments completed successfully and written to {out_dir}/.")


if __name__ == "__main__":
    run_experiments()
