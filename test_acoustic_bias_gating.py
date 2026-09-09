"""
Empirical Test: Acoustic Average Bias Multi-Gate vs. Standard Multi-Gate
Testing the hypothesis: Does adding utterance acoustic average bias improve Child WER?
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from moe_framework.production_speech_pipeline import (
    ProductionSpeechWorld,
    create_production_dataloaders,
)
from moe_framework.hybrid_moe import HybridSequenceMoE
from moe_framework.losses import DomainConditionalLoadBalanceLoss
from moe_framework.ctc_engine import SequenceErrorEvaluator, ctc_greedy_decode

DOMAINS = ["adult_speech", "child_speech", "dysarthric_speech"]


class AcousticBiasMultiGateMoE(nn.Module):
    """
    Hybrid MoE with Acoustic-Average Bias Gating:
    Computes utterance mean h_bar = (1/T) sum_t h_t, projects to bias beta_tau = W_b h_bar,
    and shifts frame routing logits: s_{t, tau} = W_g h_t + beta_tau.
    """
    def __init__(
        self,
        in_dim: int = 80,
        n_vocab: int = 27,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        hidden_dim: int = 48,
        tasks: list = DOMAINS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.n_routed_experts = n_routed_experts
        self.total_vocab = n_vocab + 1

        # 1. Frontend
        from moe_framework.sequence_models import Conv1DSubsampling
        from moe_framework.models import ExpertPool
        self.subsampling = Conv1DSubsampling(in_dim=in_dim, out_dim=hidden_dim)

        # 2. Shared & Routed Expert Pools
        self.shared_pool = ExpertPool(n_experts=n_shared_experts, in_dim=hidden_dim, hidden_dim=hidden_dim*2, out_dim=hidden_dim)
        self.routed_pool = ExpertPool(n_experts=n_routed_experts, in_dim=hidden_dim, hidden_dim=hidden_dim*2, out_dim=hidden_dim)

        # 3. Decoupled Frame Gates & Acoustic Average Bias Projectors
        self.task_gates = nn.ModuleDict({
            t: nn.Linear(hidden_dim, n_routed_experts) for t in tasks
        })
        self.bias_gates = nn.ModuleDict({
            t: nn.Linear(hidden_dim, n_routed_experts, bias=False) for t in tasks
        })

        # 4. Context & Head
        self.temporal_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.ctc_head = nn.Linear(hidden_dim, self.total_vocab)
        with torch.no_grad():
            self.ctc_head.bias[0] = -2.0

    def forward(self, x, input_lengths, task, **kwargs):
        h, sub_lengths = self.subsampling(x, input_lengths)
        B, T_sub, H = h.shape

        # Compute utterance acoustic average per batch item: (B, H)
        # Using sub_lengths to mask padding
        mask_2d = torch.arange(T_sub, device=h.device).unsqueeze(0) < sub_lengths.unsqueeze(1) # (B, T_sub)
        mask_3d = mask_2d.unsqueeze(-1).float() # (B, T_sub, 1)
        h_sum = (h * mask_3d).sum(dim=1) # (B, H)
        denom = sub_lengths.unsqueeze(-1).float().clamp(min=1.0)
        h_bar = h_sum / denom # (B, H)

        # Compute utterance-level routing bias: (B, N_experts)
        beta_speaker = self.bias_gates[task](h_bar) # (B, E)
        beta_expanded = beta_speaker.unsqueeze(1).expand(-1, T_sub, -1).reshape(B * T_sub, -1) # (B*T_sub, E)

        h_flat = h.reshape(B * T_sub, H)
        raw_logits = self.task_gates[task](h_flat)
        logits = raw_logits + beta_expanded

        G = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(G, k=self.top_k, dim=-1)
        norm_topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        Gm = torch.zeros_like(G)
        Gm.scatter_(1, topk_idx, norm_topk_vals)
        mask = torch.zeros_like(G)
        mask.scatter_(1, topk_idx, 1.0)

        routed_outs = self.routed_pool(h_flat)
        M_routed = torch.einsum("be,bed->bd", Gm, routed_outs)
        shared_outs = self.shared_pool(h_flat)
        M_shared = shared_outs.mean(dim=1)
        M_total = M_routed + M_shared

        moe_seq = M_total.reshape(B, T_sub, H)
        ctx_seq, _ = self.temporal_encoder(moe_seq)
        logits_ctc = self.ctc_head(ctx_seq)
        log_probs_ctc = F.log_softmax(logits_ctc, dim=-1).transpose(0, 1)

        return {
            "logits": logits_ctc,
            "log_probs_ctc": log_probs_ctc,
            "sub_lengths": sub_lengths,
            "Gm": Gm,
            "G": G,
            "mask": mask,
        }


def run_comparison(n_seeds: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for s in range(1, n_seeds + 1):
        cfg = {
            "in_dim": 80, "n_routed_experts": 4, "n_shared_experts": 1,
            "top_k": 2, "hidden_dim": 48, "epochs": 25, "batch_size": 25, "lr": 0.008,
            "n_train_per_domain": 250, "n_val_per_domain": 50, "n_test_per_domain": 60,
            "seed": s, "lambda_balance": 0.015,
        }
        torch.manual_seed(s)
        np.random.seed(s)

        world = ProductionSpeechWorld(cfg)
        tokenizer = world.tokenizer
        train_loaders, _, test_loaders = create_production_dataloaders(world, DOMAINS, cfg, cfg["batch_size"])

        # 1. Standard Hybrid Multi-Gate
        model_std = HybridSequenceMoE(
            in_dim=80, n_vocab=tokenizer.vocab_size - 1, n_routed_experts=4, n_shared_experts=1,
            top_k=2, hidden_dim=48, gate_type="multi_gate", tasks=DOMAINS,
        ).to(device)

        # 2. Acoustic-Average Bias Multi-Gate
        model_bias = AcousticBiasMultiGateMoE(
            in_dim=80, n_vocab=tokenizer.vocab_size - 1, n_routed_experts=4, n_shared_experts=1,
            top_k=2, hidden_dim=48, tasks=DOMAINS,
        ).to(device)

        for name, model in [("Standard Multi-Gate", model_std), ("Acoustic-Average Bias Multi-Gate", model_bias)]:
            opt = optim.Adam(model.parameters(), lr=cfg["lr"])
            ctc_fn = nn.CTCLoss(blank=0, zero_infinity=True)
            bal_fn = DomainConditionalLoadBalanceLoss(lambda_balance=cfg["lambda_balance"])

            model.train()
            min_b = min(len(train_loaders[d]) for d in DOMAINS)
            for ep in range(cfg["epochs"]):
                iters = {d: iter(train_loaders[d]) for d in DOMAINS}
                for _ in range(min_b):
                    opt.zero_grad()
                    task_G, task_M = {}, {}
                    tot_ctc = torch.tensor(0.0, device=device)
                    for d in DOMAINS:
                        b = next(iters[d])
                        out = model(b["x"].to(device), b["input_lengths"].to(device), task=d)
                        loss = ctc_fn(out["log_probs_ctc"], b["targets"].to(device), out["sub_lengths"], b["target_lengths"].to(device))
                        tot_ctc = tot_ctc + loss
                        task_G[d] = out["G"]
                        task_M[d] = out["mask"]
                    bal, _ = bal_fn(task_G, task_M, 4)
                    (tot_ctc + bal).backward()
                    opt.step()

            # Test
            model.eval()
            with torch.no_grad():
                for d in DOMAINS:
                    evaluator = SequenceErrorEvaluator(vocab_map=tokenizer.id_to_char, space_token=tokenizer.space_id)
                    for b in test_loaders[d]:
                        out = model(b["x"].to(device), b["input_lengths"].to(device), task=d)
                        hyp = ctc_greedy_decode(out["logits"], out["sub_lengths"], blank_idx=0)
                        ref = [b["targets"][i, :int(b["target_lengths"][i])].tolist() for i in range(b["targets"].size(0))]
                        evaluator.update(ref, hyp)
                    m = evaluator.compute()
                    results.append({
                        "seed": s,
                        "model": name,
                        "domain": d,
                        "WER (%)": m["WER"],
                        "CER (%)": m["CER"],
                        "sub_rate": m["word_sub_rate"],
                        "del_rate": m["word_del_rate"],
                    })

    df = pd.DataFrame(results)
    summary = df.groupby(["model", "domain"])[["WER (%)", "CER (%)", "sub_rate", "del_rate"]].agg(["mean", "std"]).round(2)
    print("\n==========================================================================================")
    print("      ACOUSTIC AVERAGE BIAS MULTI-GATE vs. STANDARD MULTI-GATE (5 SEEDS)                  ")
    print("==========================================================================================")
    print(summary.to_string())
    return df


if __name__ == "__main__":
    run_comparison(n_seeds=5)
