# Multi-gate MoE test suite

Tests the "shared expert pool + task-specific gates" architecture from the
idea repo against a single-shared-gate baseline. Pure numpy + pandas -
**no torch, no GPU, no downloads.** Runs in a few seconds on a laptop.

## Install & run

```bash
pip install numpy pandas
python moe_test_suite.py
```

That trains both models once and writes CSVs to `results/`.

To also check whether the clustering hypothesis holds up across multiple
random seeds (recommended - see "Honest caveat" below):

```bash
python moe_test_suite.py --seed-sensitivity 1 2 3 4 5 6
```

Options:
- `--out DIR` — change output directory (default `results/`)
- `--seed N` — override the single main run's seed
- `--seed-sensitivity [seeds...]` — run the full pipeline across several
  seeds and aggregate whether the hypothesis held each time

## What it's actually testing

Three synthetic downstream "tasks" (standing in for the ASR case: language
ASR, child speech, atypical speech) share a class label space. Each class is
represented in two feature blocks:

- **Block A ("phone identity")**: corrupted by a task-specific rotation.
  Language gets a *strong, distinct* rotation (large phoneme-inventory
  shift). Child and atypical share the *same mild* rotation (phone identity
  mostly intact for both).
- **Block B ("prosody / timing deviation")**: corrupted by an additive warp.
  Language gets ~no warp. Child and atypical share the *same warp
  direction*, different magnitude (both deviate from "typical adult" timing,
  for different underlying reasons - matching the idea-repo hypothesis).

If multi-gate MoE actually works the way the idea repo predicted: the
language gate should learn to lean on experts that specialize in undoing
block A's rotation, while child/atypical gates should converge on experts
that undo block B's shared warp - i.e. **child and atypical gate usage
should look similar to each other, and different from language.** That's
the testable claim `hypothesis_check.csv` checks directly.

Routing is **top-2 sparse** (Switch-Transformer style), not a dense soft
blend over all experts — this matters. An earlier dense-mixture version of
this suite showed gates staying near-uniform regardless of task, because
with enough experts a dense blend lets redundant experts co-adapt with no
gradient pressure to specialize. Sparse top-k routing is what actually
produces the specialization the hypothesis depends on.

## Output files (all in `results/`)

| File | What's in it |
|---|---|
| `training_curves.csv` | epoch-by-epoch train/val loss & accuracy, per model (`multi_gate`/`single_gate`) and per task |
| `expert_usage.csv` | per expert, per gate: mean dispatch weight and top-1 selection share, plus a row with dense-softmax entropy/perplexity |
| `gate_similarity.csv` | cosine similarity between every pair of task gates' mean expert-usage vectors |
| `hypothesis_check.csv` | the direct test: child↔atypical similarity vs. average similarity to language, and whether the hypothesis held |
| `hypothesis_check_across_seeds.csv` | (only if you ran `--seed-sensitivity`) the same check repeated across seeds, so you can see the hit rate rather than one lucky/unlucky run |
| `summary.csv` | final + best validation accuracy/loss, multi-gate vs single-gate, per task |
| `config.csv` | every hyperparameter used for this run (for reproducibility) |

## Honest caveat (read before you trust one run)

In testing, the clustering hypothesis held in **5 of 6 seeds** — a real,
mostly-consistent effect, not guaranteed every time. And the accuracy gap
between single-gate and multi-gate is **small and inconsistent**
(single-gate sometimes matches or slightly beats multi-gate on validation
accuracy), even though multi-gate consistently trains to lower *training*
loss. That's likely because in this synthetic setup, the single gate can
partially infer which task it's looking at directly from the input's
statistical pattern, even without an explicit task label — so it isn't as
handicapped as the "one gate must compress everything" story suggests at
this small scale. **The routing/clustering diagnostic is the more sensitive
signal here, not raw accuracy** — worth keeping in mind if you extend this
to real audio features, where the effect might be stronger, weaker, or
absent depending on how separable the tasks' input distributions really are.

## Tuning it further

All knobs are in the `CONFIG` dict at the top of `moe_test_suite.py`:
`n_experts`, `top_k`, `hidden`, `lambda_balance` (load-balancing strength),
`alpha_mild`/`alpha_strong` (how different language's rotation is from
child/atypical's), `mag_child`/`mag_atypical` (shared warp magnitude), and
`noise_std`/`class_scale` (task difficulty). If you push `lambda_balance`
too high, gates collapse back toward uniform and the clustering signal
disappears — that's a real failure mode worth knowing about, not just a
bug, since it's exactly the "over-balancing washes out specialization"
risk from the design doc.

## Swapping in real data later

Right now `sample_task()` generates synthetic features. To plug in TinyVox /
Vaani / Vaani-Atypical instead: replace `sample_task()` with a loader that
returns `(X, y)` — a feature matrix (e.g. MFCC or wav2vec2 embeddings
averaged per utterance) and integer phoneme/class labels — per task. Nothing
else in the training loop needs to change.
