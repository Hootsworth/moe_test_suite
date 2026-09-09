"""
Multi-gate Mixture-of-Experts test suite
=========================================
Tests the "shared expert pool + task-specific gates" architecture against
a single-shared-gate baseline, on a synthetic ASR-flavoured task with three
downstream conditions: language ASR, child speech, atypical speech.

Pure numpy (no torch, no GPU, no downloads) - runs on a laptop in seconds.

Synthetic data design (why it's built this way):
Each sample is a class y in {0..K-1} represented in two feature blocks:
  - Block A ("phone identity"): corrupted by a task-specific rotation.
    Language uses a STRONG, distinct rotation (large phoneme-inventory shift).
    Child and atypical use the SAME mild rotation (phone identity mostly intact).
  - Block B ("prosody / timing deviation"): corrupted by an additive warp.
    Language has ~no warp. Child and atypical share the same warp DIRECTION
    (different magnitude) - modeling "both deviate from typical-adult timing".

If multi-gate MoE works, the language gate should learn to lean on experts
that specialize in undoing block A's strong rotation, while the child and
atypical gates should converge on experts that specialize in undoing block
B's shared warp - i.e. child/atypical gates should look similar to each
other and different from the language gate. That's the testable hypothesis.
"""

import numpy as np
import pandas as pd
import json
import os
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = dict(
    seed=7,
    n_classes=12,
    dim_a=10,          # "phone identity" block
    dim_b=10,          # "prosody / deviation" block
    n_experts=5,
    top_k=2,
    hidden=24,
    n_train_per_task=800,
    n_val_per_task=400,
    epochs=35,
    steps_per_epoch=15,
    batch_size=40,
    lr=0.01,
    lambda_balance=0.02,
    noise_std=1.05,
    class_scale=1.4,
    alpha_mild=0.15,     # rotation strength for child/atypical (block A)
    alpha_strong=0.8,    # rotation strength for language (block A)
    mag_child=1.0,       # warp magnitude, block B
    mag_atypical=1.25,
    mag_language=0.0,
)
TASKS = ["language", "child", "atypical"]

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
def random_orthogonal(d, rng):
    m = rng.normal(0, 1, size=(d, d))
    q, r = np.linalg.qr(m)
    q *= np.sign(np.diag(r))
    return q


def build_world(cfg):
    rng = np.random.default_rng(cfg["seed"])
    K, dA, dB = cfg["n_classes"], cfg["dim_a"], cfg["dim_b"]
    base_A = rng.normal(0, 1, size=(K, dA)) * cfg["class_scale"]
    base_B = rng.normal(0, 1, size=(K, dB)) * cfg["class_scale"]
    I_A = np.eye(dA)
    Q_mild = random_orthogonal(dA, rng)
    Q_strong = random_orthogonal(dA, rng)
    C_mild = (1 - cfg["alpha_mild"]) * I_A + cfg["alpha_mild"] * Q_mild
    C_strong = (1 - cfg["alpha_strong"]) * I_A + cfg["alpha_strong"] * Q_strong
    w = rng.normal(0, 1, size=(dB,))
    w /= np.linalg.norm(w)
    task_C = {"language": C_strong, "child": C_mild, "atypical": C_mild}
    task_mag = {"language": cfg["mag_language"], "child": cfg["mag_child"], "atypical": cfg["mag_atypical"]}
    return dict(rng=rng, base_A=base_A, base_B=base_B, w=w, task_C=task_C, task_mag=task_mag)


def sample_task(world, task, n, cfg, rng=None):
    rng = rng or world["rng"]
    K = cfg["n_classes"]
    y = rng.integers(0, K, size=n)
    C = world["task_C"][task]
    mag = world["task_mag"][task]
    A = world["base_A"][y] @ C + rng.normal(0, cfg["noise_std"], size=(n, cfg["dim_a"]))
    B = world["base_B"][y] + mag * world["w"][None, :] + rng.normal(0, cfg["noise_std"], size=(n, cfg["dim_b"]))
    X = np.concatenate([A, B], axis=1)
    return X.astype(np.float64), y.astype(np.int64)


# ---------------------------------------------------------------------------
# Parameter init
# ---------------------------------------------------------------------------
def init_expert(D, H, rng):
    return dict(
        We1=rng.normal(0, np.sqrt(2.0 / D), size=(D, H)),
        be1=np.zeros(H),
        We2=rng.normal(0, np.sqrt(2.0 / H), size=(H, D)),
        be2=np.zeros(D),
    )


def init_gate(D, E, rng):
    return dict(Wg=rng.normal(0, 0.05, size=(D, E)), bg=np.zeros(E))


def init_classifier(D, K, rng):
    return dict(Wc=rng.normal(0, np.sqrt(2.0 / D), size=(D, K)), bc=np.zeros(K))


# ---------------------------------------------------------------------------
# Forward / backward (fully vectorized, manual gradients - no autodiff)
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def topk_mask(G, k):
    """Zero out all but the top-k gate weights per row (no renormalization -
    matches Switch/GShard-style scaling, where the dispatched output is
    scaled by the expert's own softmax probability)."""
    B, E = G.shape
    if k >= E:
        return np.ones_like(G)
    idx = np.argpartition(-G, k, axis=1)[:, :k]
    mask = np.zeros_like(G)
    rows = np.repeat(np.arange(B), k)
    mask[rows, idx.flatten()] = 1.0
    return mask


def forward(X, experts, gate, clf, top_k=None):
    B = X.shape[0]
    E = len(experts)
    Z1s, H1s, Outs = [], [], []
    for ex in experts:
        z1 = X @ ex["We1"] + ex["be1"]
        h1 = np.maximum(z1, 0)
        out = h1 @ ex["We2"] + ex["be2"]
        Z1s.append(z1); H1s.append(h1); Outs.append(out)
    OutStack = np.stack(Outs, axis=1)          # (B,E,D)
    Zg = X @ gate["Wg"] + gate["bg"]            # (B,E)
    G = softmax(Zg)                             # (B,E) dense softmax (used for backprop)
    if top_k is not None:
        mask = topk_mask(G, top_k)
    else:
        mask = np.ones_like(G)
    Gm = G * mask                                # (B,E) sparse dispatch weights actually used
    M = np.einsum("be,bed->bd", Gm, OutStack)    # (B,D)
    Logits = M @ clf["Wc"] + clf["bc"]           # (B,K)
    Probs = softmax(Logits)
    cache = dict(X=X, Z1s=Z1s, H1s=H1s, Outs=Outs, OutStack=OutStack,
                 Zg=Zg, G=G, Gm=Gm, mask=mask, M=M, Logits=Logits, Probs=Probs)
    return cache


def loss_and_acc(cache, y):
    B = cache["Probs"].shape[0]
    p = cache["Probs"][np.arange(B), y]
    loss = -np.mean(np.log(np.clip(p, 1e-9, 1.0)))
    acc = np.mean(np.argmax(cache["Probs"], axis=1) == y)
    return loss, acc


def backward(cache, y, experts, gate, clf, extra_dG=None):
    X, G, Gm, mask, M, Probs = cache["X"], cache["G"], cache["Gm"], cache["mask"], cache["M"], cache["Probs"]
    OutStack, Outs, H1s, Z1s = cache["OutStack"], cache["Outs"], cache["H1s"], cache["Z1s"]
    B, E = G.shape
    K = Probs.shape[1]

    onehot = np.zeros_like(Probs)
    onehot[np.arange(B), y] = 1.0
    dLogits = (Probs - onehot) / B                      # (B,K)
    dWc = M.T @ dLogits
    dbc = dLogits.sum(axis=0)
    dM = dLogits @ clf["Wc"].T                           # (B,D)

    # M was built from Gm = G*mask, so dL/dG (pre-softmax-jacobian) is zero
    # for any expert not selected this sample - that's what forces real
    # specialization instead of every expert getting a little gradient always.
    dG = mask * np.einsum("bd,bed->be", dM, OutStack)    # (B,E) from classification path
    if extra_dG is not None:
        dG = dG + extra_dG

    dZg = G * (dG - (G * dG).sum(axis=1, keepdims=True))
    dWg = X.T @ dZg
    dbg = dZg.sum(axis=0)

    grads_experts = []
    for i in range(E):
        dOut_i = Gm[:, i:i + 1] * dM                      # (B,D), zero where not selected
        dWe2 = H1s[i].T @ dOut_i
        dbe2 = dOut_i.sum(axis=0)
        dH1 = dOut_i @ experts[i]["We2"].T
        dZ1 = dH1 * (Z1s[i] > 0)
        dWe1 = X.T @ dZ1
        dbe1 = dZ1.sum(axis=0)
        grads_experts.append(dict(We1=dWe1, be1=dbe1, We2=dWe2, be2=dbe2))

    grads_gate = dict(Wg=dWg, bg=dbg)
    grads_clf = dict(Wc=dWc, bc=dbc)
    return grads_experts, grads_gate, grads_clf


def balance_aux_grad(G_list, mask_list, n_experts, lam):
    """Joint (Switch-Transformer style) load-balancing term across one or more
    gates' batches this step: L_bal = n_experts * sum_e f_e * P_e, where f_e is
    the (stop-gradient) fraction of samples dispatched to expert e and P_e is
    the mean softmax probability assigned to expert e. Minimized when both
    usage fraction and confidence are spread evenly across experts - this is
    what actually works with sparse top-k routing (the old dense mean^2 term
    just pushed everything toward uniform blending)."""
    all_G = np.concatenate(G_list, axis=0)
    all_mask = np.concatenate(mask_list, axis=0)
    N_total = all_G.shape[0]
    f_e = all_mask.mean(axis=0)                           # (E,) usage fraction, treated as constant
    d_P = lam * n_experts * f_e / N_total                  # dL/dG_e per sample (broadcast)
    aux_loss = float(n_experts * np.sum(f_e * all_G.mean(axis=0)))
    grads = [np.tile(d_P, (g.shape[0], 1)) for g in G_list]
    return grads, aux_loss


# ---------------------------------------------------------------------------
# Adam optimizer (generic, dict-of-arrays)
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads, key_prefix=""):
        self.t += 1
        for k in params:
            key = key_prefix + k
            if key not in self.m:
                self.m[key] = np.zeros_like(params[k])
                self.v[key] = np.zeros_like(params[k])
            self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * grads[k]
            self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (grads[k] ** 2)
            mhat = self.m[key] / (1 - self.b1 ** self.t)
            vhat = self.v[key] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ---------------------------------------------------------------------------
# Model containers
# ---------------------------------------------------------------------------
def build_multi_gate_model(cfg, rng):
    D = cfg["dim_a"] + cfg["dim_b"]
    experts = [init_expert(D, cfg["hidden"], rng) for _ in range(cfg["n_experts"])]
    gates = {t: init_gate(D, cfg["n_experts"], rng) for t in TASKS}
    clf = init_classifier(D, cfg["n_classes"], rng)
    return dict(experts=experts, gates=gates, clf=clf)


def build_single_gate_model(cfg, rng):
    D = cfg["dim_a"] + cfg["dim_b"]
    experts = [init_expert(D, cfg["hidden"], rng) for _ in range(cfg["n_experts"])]
    gate = init_gate(D, cfg["n_experts"], rng)
    clf = init_classifier(D, cfg["n_classes"], rng)
    return dict(experts=experts, gate=gate, clf=clf)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def train_multi_gate(cfg, world, val_data, log):
    rng = np.random.default_rng(cfg["seed"] + 1)
    model = build_multi_gate_model(cfg, rng)
    opt_experts = Adam(lr=cfg["lr"])
    opt_gates = {t: Adam(lr=cfg["lr"]) for t in TASKS}
    opt_clf = Adam(lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        ep_losses, ep_accs = {t: [] for t in TASKS}, {t: [] for t in TASKS}
        for step in range(cfg["steps_per_epoch"]):
            caches, ys, Gs, masks = {}, {}, [], []
            for t in TASKS:
                X, y = sample_task(world, t, cfg["batch_size"], cfg, rng)
                cache = forward(X, model["experts"], model["gates"][t], model["clf"], top_k=cfg["top_k"])
                caches[t] = cache
                ys[t] = y
                Gs.append(cache["G"])
                masks.append(cache["mask"])
            bal_grads, aux_loss = balance_aux_grad(Gs, masks, cfg["n_experts"], cfg["lambda_balance"])

            expert_grad_accum = None
            for i, t in enumerate(TASKS):
                l, a = loss_and_acc(caches[t], ys[t])
                ep_losses[t].append(l); ep_accs[t].append(a)
                g_experts, g_gate, g_clf = backward(caches[t], ys[t], model["experts"],
                                                     model["gates"][t], model["clf"],
                                                     extra_dG=bal_grads[i])
                opt_gates[t].step(model["gates"][t], g_gate)
                if expert_grad_accum is None:
                    expert_grad_accum = g_experts
                    clf_grad_accum = g_clf
                else:
                    for j in range(len(g_experts)):
                        for k in g_experts[j]:
                            expert_grad_accum[j][k] = expert_grad_accum[j][k] + g_experts[j][k]
                    for k in g_clf:
                        clf_grad_accum[k] = clf_grad_accum[k] + g_clf[k]

            for j, ex in enumerate(model["experts"]):
                opt_experts.step(ex, expert_grad_accum[j], key_prefix=f"e{j}_")
            opt_clf.step(model["clf"], clf_grad_accum)

        # ---- epoch-end evaluation ----
        for t in TASKS:
            Xv, yv = val_data[t]
            vcache = forward(Xv, model["experts"], model["gates"][t], model["clf"], top_k=cfg["top_k"])
            vloss, vacc = loss_and_acc(vcache, yv)
            log.append(dict(model="multi_gate", task=t, epoch=epoch,
                             train_loss=float(np.mean(ep_losses[t])),
                             train_acc=float(np.mean(ep_accs[t])),
                             val_loss=float(vloss), val_acc=float(vacc)))
    return model


def train_single_gate(cfg, world, val_data, log):
    rng = np.random.default_rng(cfg["seed"] + 2)
    model = build_single_gate_model(cfg, rng)
    opt_experts = Adam(lr=cfg["lr"])
    opt_gate = Adam(lr=cfg["lr"])
    opt_clf = Adam(lr=cfg["lr"])
    per_task_bs = max(1, cfg["batch_size"] // len(TASKS))

    for epoch in range(cfg["epochs"]):
        ep_losses, ep_accs = {t: [] for t in TASKS}, {t: [] for t in TASKS}
        for step in range(cfg["steps_per_epoch"]):
            Xs, ys, task_slices = [], [], {}
            cursor = 0
            for t in TASKS:
                X, y = sample_task(world, t, per_task_bs, cfg, rng)
                Xs.append(X); ys.append(y)
                task_slices[t] = slice(cursor, cursor + per_task_bs)
                cursor += per_task_bs
            Xb = np.concatenate(Xs, axis=0)
            yb = np.concatenate(ys, axis=0)

            cache = forward(Xb, model["experts"], model["gate"], model["clf"], top_k=cfg["top_k"])
            bal_grads, aux_loss = balance_aux_grad([cache["G"]], [cache["mask"]], cfg["n_experts"], cfg["lambda_balance"])
            g_experts, g_gate, g_clf = backward(cache, yb, model["experts"], model["gate"],
                                                 model["clf"], extra_dG=bal_grads[0])
            opt_gate.step(model["gate"], g_gate)
            for j, ex in enumerate(model["experts"]):
                opt_experts.step(ex, g_experts[j], key_prefix=f"e{j}_")
            opt_clf.step(model["clf"], g_clf)

            probs_full = cache["Probs"]
            for t in TASKS:
                sl = task_slices[t]
                p = probs_full[sl][np.arange(per_task_bs), yb[sl]]
                l = -np.mean(np.log(np.clip(p, 1e-9, 1.0)))
                a = np.mean(np.argmax(probs_full[sl], axis=1) == yb[sl])
                ep_losses[t].append(l); ep_accs[t].append(a)

        for t in TASKS:
            Xv, yv = val_data[t]
            vcache = forward(Xv, model["experts"], model["gate"], model["clf"], top_k=cfg["top_k"])
            vloss, vacc = loss_and_acc(vcache, yv)
            log.append(dict(model="single_gate", task=t, epoch=epoch,
                             train_loss=float(np.mean(ep_losses[t])),
                             train_acc=float(np.mean(ep_accs[t])),
                             val_loss=float(vloss), val_acc=float(vacc)))
    return model


# ---------------------------------------------------------------------------
# Post-training analysis
# ---------------------------------------------------------------------------
def expert_usage_stats(model_type, model, val_data, cfg):
    rows = []
    gate_vectors = {}
    for t in TASKS:
        Xv, yv = val_data[t]
        gate = model["gates"][t] if model_type == "multi_gate" else model["gate"]
        cache = forward(Xv, model["experts"], gate, model["clf"], top_k=cfg["top_k"])
        G, Gm, mask = cache["G"], cache["Gm"], cache["mask"]
        mean_w = Gm.mean(axis=0)                      # actual average dispatch weight per expert
        top1_share = mask.mean(axis=0)                # fraction of samples each expert was selected for
        gate_vectors[t] = mean_w
        entropy = -np.sum(G * np.log(np.clip(G, 1e-12, 1.0)), axis=1).mean()
        perplexity = float(np.exp(entropy))
        for e in range(cfg["n_experts"]):
            rows.append(dict(model=model_type, task=t, expert_id=e,
                              mean_gate_weight=float(mean_w[e]),
                              top1_usage_share=float(top1_share[e])))
        rows.append(dict(model=model_type, task=t, expert_id="ALL_mean_entropy",
                          mean_gate_weight=float(entropy), top1_usage_share=float(perplexity)))
        if model_type == "single_gate":
            break  # only one gate, no need to repeat identical stats per task
    return rows, gate_vectors


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def gate_similarity_table(model_type, gate_vectors):
    rows = []
    tasks = list(gate_vectors.keys())
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            ta, tb = tasks[i], tasks[j]
            sim = cosine_sim(gate_vectors[ta], gate_vectors[tb])
            rows.append(dict(model=model_type, task_a=ta, task_b=tb, cosine_similarity=sim))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(out_dir="results", seed_override=None, quiet=False):
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(CONFIG)
    if seed_override is not None:
        cfg["seed"] = seed_override
    world = build_world(cfg)

    rng_val = np.random.default_rng(cfg["seed"] + 999)
    val_data = {t: sample_task(world, t, cfg["n_val_per_task"], cfg, rng_val) for t in TASKS}

    t0 = time.time()
    log = []
    if not quiet:
        print("Training multi-gate MoE ...")
    multi_model = train_multi_gate(cfg, world, val_data, log)
    if not quiet:
        print("Training single-gate baseline ...")
    single_model = train_single_gate(cfg, world, val_data, log)
    elapsed = time.time() - t0
    if not quiet:
        print(f"Done in {elapsed:.1f}s")

    # ---- training_curves.csv ----
    df_curves = pd.DataFrame(log)
    df_curves.to_csv(os.path.join(out_dir, "training_curves.csv"), index=False)

    # ---- expert_usage.csv + gate_similarity.csv ----
    usage_rows, sim_rows = [], []
    multi_usage, multi_gates = expert_usage_stats("multi_gate", multi_model, val_data, cfg)
    single_usage, single_gates = expert_usage_stats("single_gate", single_model, val_data, cfg)
    usage_rows += multi_usage + single_usage
    sim_rows += gate_similarity_table("multi_gate", multi_gates)
    sim_rows += gate_similarity_table("single_gate", single_gates)  # trivial/degenerate but included for completeness
    pd.DataFrame(usage_rows).to_csv(os.path.join(out_dir, "expert_usage.csv"), index=False)
    pd.DataFrame(sim_rows).to_csv(os.path.join(out_dir, "gate_similarity.csv"), index=False)

    # ---- hypothesis_check.csv ----
    hyp_rows = []
    sim_df = pd.DataFrame(sim_rows)
    m = sim_df[sim_df["model"] == "multi_gate"].set_index(["task_a", "task_b"])["cosine_similarity"]

    def get_sim(a, b):
        if (a, b) in m.index:
            return m.loc[(a, b)]
        return m.loc[(b, a)]

    sim_child_atyp = get_sim("child", "atypical")
    sim_child_lang = get_sim("language", "child") if ("language", "child") in m.index else get_sim("child", "language")
    sim_atyp_lang = get_sim("language", "atypical") if ("language", "atypical") in m.index else get_sim("atypical", "language")
    avg_divergent = (sim_child_lang + sim_atyp_lang) / 2
    hyp_rows.append(dict(
        hypothesis="child & atypical gates cluster; language gate diverges",
        similarity_child_atypical=float(sim_child_atyp),
        avg_similarity_to_language=float(avg_divergent),
        margin=float(sim_child_atyp - avg_divergent),
        hypothesis_supported=bool(sim_child_atyp > avg_divergent),
    ))
    pd.DataFrame(hyp_rows).to_csv(os.path.join(out_dir, "hypothesis_check.csv"), index=False)

    # ---- summary.csv ----
    summary_rows = []
    for model_name in ["multi_gate", "single_gate"]:
        sub = df_curves[df_curves["model"] == model_name]
        final_epoch = sub["epoch"].max()
        for t in TASKS:
            s = sub[(sub["task"] == t)]
            final = s[s["epoch"] == final_epoch].iloc[0]
            best_val_acc = s["val_acc"].max()
            summary_rows.append(dict(
                model=model_name, task=t,
                final_train_loss=final["train_loss"], final_train_acc=final["train_acc"],
                final_val_loss=final["val_loss"], final_val_acc=final["val_acc"],
                best_val_acc=float(best_val_acc),
            ))
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # ---- config.csv ----
    cfg_rows = [dict(parameter=k, value=v) for k, v in cfg.items()]
    cfg_rows.append(dict(parameter="elapsed_seconds", value=round(elapsed, 2)))
    pd.DataFrame(cfg_rows).to_csv(os.path.join(out_dir, "config.csv"), index=False)

    if not quiet:
        print("\nAll CSVs written to:", os.path.abspath(out_dir))
        print(df_summary.to_string(index=False))
        print("\nHypothesis check:")
        print(pd.DataFrame(hyp_rows).to_string(index=False))

    return dict(summary=df_summary, hypothesis=hyp_rows[0], elapsed=elapsed)


def run_seed_sensitivity(seeds, base_out="results"):
    """Re-run the whole pipeline across several seeds to check whether the
    clustering hypothesis holds robustly or was a lucky/unlucky single run."""
    rows = []
    seed_dir = os.path.join(base_out, "seed_sensitivity")
    os.makedirs(seed_dir, exist_ok=True)
    for s in seeds:
        print(f"[seed sensitivity] running seed={s} ...")
        res = main(out_dir=os.path.join(seed_dir, f"seed_{s}"), seed_override=s, quiet=True)
        row = dict(seed=s)
        row.update(res["hypothesis"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base_out, "hypothesis_check_across_seeds.csv"), index=False)
    n_support = df["hypothesis_supported"].sum()
    print(f"\nHypothesis supported in {n_support}/{len(df)} seeds")
    print(df[["seed", "similarity_child_atypical", "avg_similarity_to_language", "margin", "hypothesis_supported"]].to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-gate MoE test suite")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="single seed override")
    parser.add_argument("--seed-sensitivity", nargs="*", type=int, default=None,
                         help="run across multiple seeds to check hypothesis robustness, e.g. --seed-sensitivity 1 2 3 4 5")
    args = parser.parse_args()

    main(out_dir=args.out, seed_override=args.seed)
    if args.seed_sensitivity is not None:
        seeds = args.seed_sensitivity if len(args.seed_sensitivity) > 0 else [1, 2, 3, 4, 5, 6]
        run_seed_sensitivity(seeds, base_out=args.out)
