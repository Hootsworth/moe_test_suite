from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_gini(array: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a numpy array.
    Gini = 0 represents perfect equality, Gini = 1 represents maximal specialization.
    """
    array = np.asarray(array, dtype=float).flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-12  # avoid zero division
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float(((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@torch.no_grad()
def compute_gate_statistics(
    model: torch.nn.Module,
    val_loaders: Dict[str, DataLoader],
    tasks: List[str],
    device: str = "cpu",
) -> Tuple[List[Dict[str, Union[str, int, float]]], Dict[str, np.ndarray]]:
    """
    Compute per-expert, per-domain routing statistics:
      - mean dispatch weight Gm
      - top-1 selection share
      - router entropy and perplexity
    Returns:
        rows: List of dictionary records
        gate_vectors: Dict[task -> mean_Gm_vector (E,)]
    """
    model.eval()
    rows = []
    gate_vectors = {}

    for task in tasks:
        all_Gm, all_G, all_mask = [], [], []

        for batch in val_loaders[task]:
            x = batch["x"].to(device)
            z = batch["z_acoustic"].to(device)
            out = model(x=x, task=task, acoustic_emb=z)

            all_Gm.append(out["Gm"].cpu().numpy())
            all_G.append(out["G"].cpu().numpy())
            all_mask.append(out["mask"].cpu().numpy())

        Gm_concat = np.concatenate(all_Gm, axis=0)      # (N, E)
        G_concat = np.concatenate(all_G, axis=0)        # (N, E)
        mask_concat = np.concatenate(all_mask, axis=0)  # (N, E)

        mean_w = Gm_concat.mean(axis=0)                 # (E,)
        top1_share = mask_concat.mean(axis=0)           # (E,)
        gate_vectors[task] = mean_w

        # Shannon entropy of dense gate probabilities
        clipped_G = np.clip(G_concat, 1e-12, 1.0)
        entropy = -np.sum(clipped_G * np.log(clipped_G), axis=1).mean()
        perplexity = float(np.exp(entropy))

        n_experts = Gm_concat.shape[1]
        for e in range(n_experts):
            rows.append({
                "model_gate_type": model.gate_type,
                "task": task,
                "expert_id": str(e),
                "mean_gate_weight": float(mean_w[e]),
                "usage_share": float(top1_share[e]),
            })

        rows.append({
            "model_gate_type": model.gate_type,
            "task": task,
            "expert_id": "ALL_entropy",
            "mean_gate_weight": float(entropy),
            "usage_share": float(perplexity),
        })

    return rows, gate_vectors


def compute_gate_cosine_similarity(
    gate_vectors: Dict[str, np.ndarray],
    model_gate_type: str,
) -> List[Dict[str, Union[str, float]]]:
    """Compute pairwise cosine similarities across task gate vectors."""
    rows = []
    tasks = list(gate_vectors.keys())
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            t_a, t_b = tasks[i], tasks[j]
            sim = cosine_similarity(gate_vectors[t_a], gate_vectors[t_b])
            rows.append({
                "model_gate_type": model_gate_type,
                "task_a": t_a,
                "task_b": t_b,
                "cosine_similarity": sim,
            })
    return rows


def compute_gini_specialization(
    gate_vectors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute Gini Specialization Index across domains:
    For each expert, measure how skewed its usage is across tasks.
    """
    tasks = list(gate_vectors.keys())
    n_experts = len(gate_vectors[tasks[0]])
    expert_task_matrix = np.stack([gate_vectors[t] for t in tasks], axis=1)  # (E, n_tasks)

    gini_per_expert = {}
    for e in range(n_experts):
        usage_across_tasks = expert_task_matrix[e, :]
        gini = compute_gini(usage_across_tasks)
        gini_per_expert[f"expert_{e}_gini"] = round(gini, 4)

    gini_per_expert["mean_expert_gini"] = round(float(np.mean(list(gini_per_expert.values()))), 4)
    return gini_per_expert


@torch.no_grad()
def run_expert_ablation(
    model: torch.nn.Module,
    val_loaders: Dict[str, DataLoader],
    tasks: List[str],
    device: str = "cpu",
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Systematically ablate each expert (zero out its dispatch) and record
    validation accuracy degradation across tasks.
    """
    model.eval()
    n_experts = model.n_routed_experts
    records = []

    # 1. Baseline accuracy with all experts active
    base_accs = {}
    for task in tasks:
        correct, total = 0, 0
        for batch in val_loaders[task]:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            z = batch["z_acoustic"].to(device)
            out = model(x=x, task=task, acoustic_emb=z)
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        base_accs[task] = correct / total if total > 0 else 0.0

    for task in tasks:
        records.append({
            "task": task,
            "ablated_expert": "None (Baseline)",
            "accuracy": round(base_accs[task], 4),
            "delta_acc": 0.0,
        })

    # 2. Per-expert ablation
    for e in range(n_experts):
        for task in tasks:
            correct, total = 0, 0
            for batch in val_loaders[task]:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                z = batch["z_acoustic"].to(device)
                out = model(x=x, task=task, acoustic_emb=z, ablate_expert_ids=[e])
                preds = out["logits"].argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            ablated_acc = correct / total if total > 0 else 0.0
            records.append({
                "task": task,
                "ablated_expert": f"Expert_{e}",
                "accuracy": round(ablated_acc, 4),
                "delta_acc": round(ablated_acc - base_accs[task], 4),
            })

    return records


def evaluate_hypothesis(
    sim_rows: List[Dict[str, Union[str, float]]],
) -> Dict[str, Union[str, float, bool]]:
    """
    Test the core hypothesis:
    sim(child, atypical) > avg(sim(child, language), sim(atypical, language))
    """
    sim_map = {}
    for r in sim_rows:
        pair = (r["task_a"], r["task_b"])
        sim_map[pair] = r["cosine_similarity"]
        sim_map[(r["task_b"], r["task_a"])] = r["cosine_similarity"]

    sim_child_atyp = sim_map.get(("child", "atypical"), 0.0)
    sim_child_lang = sim_map.get(("child", "language"), 0.0)
    sim_atyp_lang = sim_map.get(("atypical", "language"), 0.0)
    avg_lang_sim = (sim_child_lang + sim_atyp_lang) / 2.0
    margin = sim_child_atyp - avg_lang_sim

    return {
        "hypothesis": "child & atypical gates cluster; language gate diverges",
        "similarity_child_atypical": round(float(sim_child_atyp), 4),
        "avg_similarity_to_language": round(float(avg_lang_sim), 4),
        "margin": round(float(margin), 4),
        "hypothesis_supported": bool(sim_child_atyp > avg_lang_sim),
    }
