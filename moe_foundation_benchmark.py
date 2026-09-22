"""
Foundation Model Clinical MoE Benchmark Suite (Wav2Vec 2.0 Adapter)
===================================================================
Evaluates the Neuromotor-Decoupled Clinical MoE and Speaker
Acoustic-Average Calibration as parameter-efficient adapters on top of
frozen Self-Supervised Speech Foundation Models (WAV2VEC2_BASE):
  1. Frozen Wav2Vec2 + Linear Probe (CTC Head Baseline)
  2. Frozen Wav2Vec2 + Single-Gate MoE Adapter
  3. Frozen Wav2Vec2 + Decoupled Severity MoE Adapter
  4. Frozen Wav2Vec2 + Calibrated Clinical MoE Adapter (Ours)

Evaluated under strict speaker-disjoint cross-validation (S_train ∩ S_test = ∅).
Outputs:
  - foundation_benchmark_results.csv
  - foundation_benchmark_results.json
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
from torch.utils.data import Dataset, DataLoader
import torchaudio

from moe_framework.torgo_pipeline import (
    TORGO_SPEAKER_REGISTRY,
    TORGO_BENCHMARK_PROMPTS,
    TorgoPartitionManager,
)
from moe_framework.foundation_adapter import FoundationClinicalMoEAdapter
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
    ("linear_probe_baseline", "linear_probe", 0),
    ("single_gate_moe_adapter", "standard_single_gate", 0),
    ("decoupled_severity_adapter", "decoupled_severity", 1),
    ("calibrated_clinical_moe_adapter", "calibrated_atypical", 1),
]


def synthesize_audio_waveform(
    spk_id: str,
    text: str,
    sr: int = 16000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Synthesizes realistic 16kHz audio waveform matching clinical pathology."""
    rng = rng or np.random.default_rng(42)
    meta = TORGO_SPEAKER_REGISTRY.get(spk_id)
    sev = meta.severity if meta else "control_typical"

    if sev == "control_typical":
        vt_scale = float(rng.uniform(0.98, 1.02))
        centralization = 0.05
        tempo = float(rng.uniform(1.0, 1.1))
        glottal_noise = 0.12
    elif sev == "mild_dysarthria":
        vt_scale = float(rng.uniform(0.95, 1.05))
        centralization = 0.22
        tempo = float(rng.uniform(1.2, 1.35))
        glottal_noise = 0.28
    elif sev == "moderate_dysarthria":
        vt_scale = float(rng.uniform(0.92, 1.08))
        centralization = 0.48
        tempo = float(rng.uniform(1.5, 1.75))
        glottal_noise = 0.50
    else:  # severe_dysarthria
        vt_scale = float(rng.uniform(0.90, 1.10))
        centralization = 0.75
        tempo = float(rng.uniform(1.9, 2.3))
        glottal_noise = 0.75

    clean_text = "".join([c for c in text.upper() if c.isalnum() or c == " "])
    token_ids = [ord(c) - ord('A') + 1 if 'A' <= c <= 'Z' else 27 for c in clean_text]
    if not token_ids:
        token_ids = [1, 2, 3]

    sec_per_token = 0.08 * tempo
    total_sec = max(0.5, len(token_ids) * sec_per_token)
    total_samples = int(sr * total_sec)
    t = np.linspace(0, total_sec, total_samples, endpoint=False)

    f0 = 120.0 if (meta and meta.gender == "M") else 210.0
    f0 = f0 * (1.0 + rng.normal(0, 0.03))

    wave = np.zeros(total_samples, dtype=np.float32)
    schwa_f1, schwa_f2 = 500.0, 1500.0

    samples_per_tok = total_samples // len(token_ids)
    for i, tok in enumerate(token_ids):
        s_start = i * samples_per_tok
        s_end = min(total_samples, (i + 1) * samples_per_tok)
        if s_start >= s_end:
            continue
        tok_t = t[s_start:s_end]

        # Formant frequencies modulated by speaker VTLN and vowel centralization
        canon_f1 = 300.0 + (tok * 37) % 550
        canon_f2 = 900.0 + (tok * 73) % 1500
        f1 = (1.0 - centralization) * (canon_f1 * vt_scale) + centralization * schwa_f1
        f2 = (1.0 - centralization) * (canon_f2 * vt_scale) + centralization * schwa_f2

        harmonic = (
            0.5 * np.sin(2 * np.pi * f0 * tok_t)
            + 0.3 * np.sin(2 * np.pi * f1 * tok_t)
            + 0.2 * np.sin(2 * np.pi * f2 * tok_t)
        )
        noise = rng.normal(0, glottal_noise, size=len(tok_t))
        wave[s_start:s_end] = harmonic * (1.0 - glottal_noise * 0.4) + noise * 0.4

    # Peak normalization
    max_val = np.max(np.abs(wave)) + 1e-6
    wave = wave / max_val * 0.9
    return torch.from_numpy(wave).float(), token_ids


class CachedFoundationDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_cached_samples(batch: List[Dict]) -> Dict:
    feats = [b["features"] for b in batch]
    feat_lens = torch.tensor([b["feat_length"] for b in batch], dtype=torch.long)
    targets = [b["target"] for b in batch]
    tar_lens = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

    max_t = int(feat_lens.max())
    d_feat = feats[0].shape[-1]
    padded_feats = torch.zeros(len(feats), max_t, d_feat, dtype=torch.float32)
    for i, f in enumerate(feats):
        padded_feats[i, :f.shape[0], :] = f

    max_u = max(1, int(tar_lens.max()))
    padded_targets = torch.zeros(len(targets), max_u, dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :t.shape[0]] = t

    return {
        "features": padded_feats,
        "feat_lengths": feat_lens,
        "targets": padded_targets,
        "target_lengths": tar_lens,
        "severities": [b["severity"] for b in batch],
        "speaker_ids": [b["speaker_id"] for b in batch],
        "transcripts": [b["transcript"] for b in batch],
    }


def precompute_foundation_features(
    speaker_ids: List[str],
    n_utts_per_speaker: int,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Passes audio waveforms through frozen WAV2VEC2_BASE once and caches representations."""
    print(f"Loading frozen WAV2VEC2_BASE backbone to extract contextual representations...")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    backbone = bundle.get_model().eval()
    rng = np.random.default_rng(seed)

    by_sev = {s: [] for s in SEVERITY_LEVELS}

    for spk_id in speaker_ids:
        meta = TORGO_SPEAKER_REGISTRY.get(spk_id)
        if not meta:
            continue
        sev = meta.severity

        for u_idx in range(n_utts_per_speaker):
            prompt = TORGO_BENCHMARK_PROMPTS[u_idx % len(TORGO_BENCHMARK_PROMPTS)]
            words = prompt.split()
            if len(words) > 4:
                w_len = rng.integers(3, 6)
                start_w = rng.integers(0, max(1, len(words) - w_len + 1))
                text = " ".join(words[start_w:start_w + w_len])
            else:
                text = prompt

            wave, tokens = synthesize_audio_waveform(spk_id, text, rng=rng)
            wave_tensor = wave.unsqueeze(0)  # (1, L)
            wave_len = torch.tensor([wave.shape[0]])

            with torch.no_grad():
                feat_seq, out_len = backbone(wave_tensor, wave_len)

            item = {
                "features": feat_seq.squeeze(0),  # (T, 768)
                "feat_length": int(out_len.item()),
                "target": torch.tensor(tokens, dtype=torch.long),
                "severity": sev,
                "speaker_id": spk_id,
                "transcript": text,
            }
            by_sev[sev].append(item)

    return by_sev


def train_adapter_epoch(model, loaders, optimizer, ctc_fn, bal_fn, device):
    model.train()
    min_b = min(len(loaders[s]) for s in SEVERITY_LEVELS)
    iters = {s: iter(loaders[s]) for s in SEVERITY_LEVELS}

    for _ in range(min_b):
        optimizer.zero_grad()
        task_G, task_M = {}, {}
        total_ctc = torch.tensor(0.0, device=device)

        for sev in SEVERITY_LEVELS:
            batch = next(iters[sev])
            feats = batch["features"].to(device)
            f_lens = batch["feat_lengths"].to(device)
            targets = batch["targets"].to(device)
            t_lens = batch["target_lengths"].to(device)

            out = model(features=feats, lengths=f_lens, severity=sev)
            loss = ctc_fn(out["log_probs_ctc"], targets, out["out_lengths"], t_lens)
            total_ctc = total_ctc + loss

            if model.routing_mode != "linear_probe":
                task_G[sev] = out["G"]
                task_M[sev] = out["mask"]

        bal = torch.tensor(0.0, device=device)
        if bal_fn is not None and task_G and model.routing_mode != "linear_probe":
            bal, _ = bal_fn(task_G, task_M, model.n_routed_experts)

        (total_ctc + bal).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def evaluate_adapter_cohort(model, loaders, tokenizer, device):
    model.eval()
    results = {}

    for sev in SEVERITY_LEVELS:
        evaluator = SequenceErrorEvaluator()
        loader = loaders[sev]

        with torch.no_grad():
            for batch in loader:
                feats = batch["features"].to(device)
                f_lens = batch["feat_lengths"].to(device)
                targets = batch["targets"]
                t_lens = batch["target_lengths"]

                out = model(features=feats, lengths=f_lens, severity=sev)
                hyps = ctc_greedy_decode(out["logits"], out["out_lengths"], blank_idx=0)

                refs = []
                for b in range(targets.size(0)):
                    u = int(t_lens[b].item())
                    refs.append(targets[b, :u].cpu().tolist())

                evaluator.update(refs, hyps)

        summary = evaluator.compute()
        results[sev] = summary

    macro_cer = float(np.mean([results[s]["CER"] for s in SEVERITY_LEVELS]))
    macro_wer = float(np.mean([results[s]["WER"] for s in SEVERITY_LEVELS]))
    results["macro"] = {"CER": round(macro_cer, 2), "WER": round(macro_wer, 2)}
    return results


def run_benchmark(epochs: int = 20, seed: int = 42):
    train_spks, test_spks = TorgoPartitionManager.get_canonical_disjoint_split()
    print("=" * 80)
    print("FOUNDATION MODEL CLINICAL MOE ADAPTER BENCHMARK (Wav2Vec 2.0)")
    print("=" * 80)
    print(f"Train Speakers ({len(train_spks)}): {', '.join(train_spks)}")
    print(f"Test Speakers  ({len(test_spks)}):  {', '.join(test_spks)}")
    print("-" * 80)

    # 1. Precompute cached representations
    t0_pre = time.time()
    train_feats = precompute_foundation_features(train_spks, n_utts_per_speaker=30, seed=seed)
    test_feats = precompute_foundation_features(test_spks, n_utts_per_speaker=18, seed=seed + 999)
    print(f"Representation pre-extraction completed in {time.time() - t0_pre:.2f}s.")

    train_loaders = {}
    test_loaders = {}
    for s in SEVERITY_LEVELS:
        train_loaders[s] = DataLoader(
            CachedFoundationDataset(train_feats[s]),
            batch_size=16,
            shuffle=True,
            collate_fn=collate_cached_samples,
        )
        test_loaders[s] = DataLoader(
            CachedFoundationDataset(test_feats[s]),
            batch_size=16,
            shuffle=False,
            collate_fn=collate_cached_samples,
        )

    tokenizer = EnglishPhoneticTokenizer()
    device = torch.device("cpu")
    ctc_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    bal_fn = DomainConditionalLoadBalanceLoss(lambda_balance=0.01)

    all_results = []

    for name, mode, n_shared in MODELS:
        print(f"\nEvaluating: {name.upper()} (mode={mode}, n_shared={n_shared})")
        torch.manual_seed(seed)

        model = FoundationClinicalMoEAdapter(
            backbone_name="WAV2VEC2_BASE",
            adapter_dim=64,
            n_vocab=tokenizer.vocab_size - 1,
            n_routed_experts=4,
            n_shared_experts=n_shared,
            top_k=2,
            routing_mode=mode,
            freeze_backbone=True,
        ).to(device)

        counts = model.count_parameters()
        print(
            f"  Trainable Adapter: {counts['trainable_adapter']:,} params | "
            f"Frozen Backbone: {counts['total_backbone']:,} params | "
            f"Trainable Ratio: {counts['trainable_pct']:.3f}%"
        )

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.003,
            weight_decay=1e-4,
        )

        t0 = time.time()
        for epoch in range(epochs):
            train_adapter_epoch(model, train_loaders, optimizer, ctc_fn, bal_fn, device)
        dur = time.time() - t0

        eval_res = evaluate_adapter_cohort(model, test_loaders, tokenizer, device)

        row = {
            "model": name,
            "routing_mode": mode,
            "trainable_params": counts["trainable_adapter"],
            "param_pct": round(counts["trainable_pct"], 3),
            "train_sec": round(dur, 2),
            "control_cer": eval_res["control_typical"]["CER"],
            "mild_cer": eval_res["mild_dysarthria"]["CER"],
            "moderate_cer": eval_res["moderate_dysarthria"]["CER"],
            "severe_cer": eval_res["severe_dysarthria"]["CER"],
            "macro_cer": eval_res["macro"]["CER"],
            "control_wer": eval_res["control_typical"]["WER"],
            "mild_wer": eval_res["mild_dysarthria"]["WER"],
            "moderate_wer": eval_res["moderate_dysarthria"]["WER"],
            "severe_wer": eval_res["severe_dysarthria"]["WER"],
            "macro_wer": eval_res["macro"]["WER"],
        }
        all_results.append(row)

        print(
            f"  -> Macro CER: {row['macro_cer']}% | Control: {row['control_cer']}% | "
            f"Mild: {row['mild_cer']}% | Mod: {row['moderate_cer']}% | Sev: {row['severe_cer']}%"
        )

    df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("FINAL FOUNDATION MODEL BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    cols = ["model", "trainable_params", "param_pct", "control_cer", "mild_cer", "moderate_cer", "severe_cer", "macro_cer"]
    print(df[cols].to_string(index=False))
    print("=" * 80)

    csv_path = "foundation_benchmark_results.csv"
    json_path = "foundation_benchmark_results.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark outputs to {csv_path} and {json_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(epochs=args.epochs, seed=args.seed)
