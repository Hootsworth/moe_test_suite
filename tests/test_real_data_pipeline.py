"""
Unit tests for Real-World Audio and Project Vaani Data Pipeline
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from moe_framework.audio_features import LogMelSpectrogramExtractor, SpecAugment, extract_acoustic_signature
from moe_framework.real_data_pipeline import DevanagariTokenizer, VaaniSpeechWorld, create_vaani_dataloaders
from moe_framework.sequence_models import SequenceMultiGateMoE


def test_log_mel_extractor():
    # 1 second of dummy 16kHz audio
    waveform = torch.sin(2 * np.pi * 440 * torch.linspace(0, 1, 16000))
    extractor = LogMelSpectrogramExtractor(sr=16000, n_mels=80)
    mel = extractor(waveform)  # (1, T, 80)

    assert mel.ndim == 3
    assert mel.size(2) == 80
    assert mel.size(1) > 90  # ~100 frames for 1s


def test_devanagari_tokenizer():
    tokenizer = DevanagariTokenizer()
    text = "आज का मौसम बहुत अच्छा है"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert len(tokens) > 0
    assert decoded == text
    assert tokenizer.blank_id == 0


def test_vaani_world_and_dataloader():
    cfg = {"seed": 42, "n_train_real_per_domain": 10, "n_val_real_per_domain": 5}
    world = VaaniSpeechWorld(cfg)
    domains = ["standard_hindi", "bhojpuri_dialect", "rural_field_atypical"]

    train_loaders, val_loaders = create_vaani_dataloaders(world, domains, cfg, batch_size=4)
    batch = next(iter(train_loaders["bhojpuri_dialect"]))

    assert batch["x"].size(2) == 80  # 80-dim log-mel
    assert batch["targets"].ndim == 2
    assert batch["input_lengths"].size(0) == 4
    assert batch["z_acoustic"].size(1) == 4


def test_real_speech_moe_forward():
    tokenizer = DevanagariTokenizer()
    model = SequenceMultiGateMoE(
        in_dim=80,
        n_vocab=tokenizer.vocab_size - 1,
        n_routed_experts=5,
        top_k=2,
        hidden_dim=48,
        gate_type="multi_gate",
        tasks=["standard_hindi", "bhojpuri_dialect", "rural_field_atypical"],
    )

    B, T_raw, D = 4, 120, 80
    x = torch.randn(B, T_raw, D)
    in_lens = torch.full((B,), T_raw, dtype=torch.long)
    z = torch.randn(B, 4)

    out = model(x=x, input_lengths=in_lens, task="bhojpuri_dialect", acoustic_emb=z)
    assert out["logits"].size(2) == tokenizer.vocab_size
    assert out["sub_lengths"].size(0) == B


if __name__ == "__main__":
    test_log_mel_extractor()
    test_devanagari_tokenizer()
    test_vaani_world_and_dataloader()
    test_real_speech_moe_forward()
    print("All Real Speech and Vaani pipeline unit tests passed successfully!")
