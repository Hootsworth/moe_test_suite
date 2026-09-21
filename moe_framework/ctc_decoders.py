"""
Advanced CTC Decoding Suite for Clinical and Dysarthric Speech Recognition
=========================================================================
Implements:
  1. Greedy CTC with Blank Penalty (gamma)
  2. CTC Prefix Beam Search
  3. Character / Word N-Gram Language Model Integration
  4. Lexicon-Constrained Beam Search Decoding
"""

from collections import defaultdict
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


class SimpleNGramLM:
    """
    Word-level and Character-level smoothed N-Gram Language Model.
    Trained on in-domain corpus text with Add-k Laplace smoothing.
    """
    def __init__(self, n: int = 3, smoothing: float = 0.01):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab: Set[str] = set()

    def train(self, sentences: List[str]):
        for sent in sentences:
            words = ["<s>"] + sent.upper().split() + ["</s>"]
            for w in words:
                self.vocab.add(w)
            for i in range(len(words)):
                for order in range(1, self.n + 1):
                    if i - order + 1 >= 0:
                        ngram = tuple(words[i - order + 1 : i + 1])
                        ctx = ngram[:-1]
                        self.counts[ngram] += 1
                        self.context_counts[ctx] += 1

    def score_sentence(self, sentence: str) -> float:
        words = ["<s>"] + sentence.upper().split() + ["</s>"]
        log_prob = 0.0
        v_size = max(1, len(self.vocab))

        for i in range(1, len(words)):
            ctx = tuple(words[max(0, i - self.n + 1) : i])
            ngram = ctx + (words[i],)
            count = self.counts.get(ngram, 0)
            ctx_count = self.context_counts.get(ctx, 0)
            p = (count + self.smoothing) / (ctx_count + self.smoothing * v_size)
            log_prob += math.log(p)

        return log_prob

    def score_word_transition(self, prev_words: List[str], next_word: str) -> float:
        ctx = tuple((["<s>"] + prev_words)[-self.n + 1 :])
        ngram = ctx + (next_word,)
        v_size = max(1, len(self.vocab))
        count = self.counts.get(ngram, 0)
        ctx_count = self.context_counts.get(ctx, 0)
        p = (count + self.smoothing) / (ctx_count + self.smoothing * v_size)
        return math.log(p)


def ctc_greedy_decode_with_blank_penalty(
    logits: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    blank_idx: int = 0,
    blank_penalty: float = 0.0,
) -> List[List[int]]:
    """
    Greedy CTC argmax decoding with a blank penalty gamma:
      log p'(blank) = log p(blank) - gamma
    Downweights the blank token logit to prevent severe dysarthric deletion collapse.
    """
    if logits.ndim == 3 and input_lengths is not None and logits.size(0) != input_lengths.size(0):
        logits = logits.transpose(0, 1)

    log_probs = F.log_softmax(logits, dim=-1).clone()

    if blank_penalty > 0.0:
        log_probs[:, :, blank_idx] -= blank_penalty

    preds = log_probs.argmax(dim=-1).cpu().numpy()
    B, max_t = preds.shape
    decoded_seqs = []

    for b in range(B):
        t_len = int(input_lengths[b].item()) if input_lengths is not None else max_t
        seq = preds[b, :t_len]
        collapsed = []
        prev = None
        for tok in seq:
            if tok != prev:
                if tok != blank_idx:
                    collapsed.append(int(tok))
                prev = tok
        decoded_seqs.append(collapsed)

    return decoded_seqs


def ctc_prefix_beam_search(
    logits: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    vocab_map: Optional[Dict[int, str]] = None,
    beam_width: int = 10,
    blank_idx: int = 0,
    blank_penalty: float = 0.0,
    lm: Optional[SimpleNGramLM] = None,
    lm_weight: float = 0.5,
    lexicon: Optional[Set[str]] = None,
    word_bonus: float = 0.1,
) -> List[List[int]]:
    """
    CTC Prefix Beam Search Decoder with optional Blank Penalty, N-Gram LM, and Lexicon filtering.
    """
    if logits.ndim == 3 and input_lengths is not None and logits.size(0) != input_lengths.size(0):
        logits = logits.transpose(0, 1)

    log_probs = F.log_softmax(logits, dim=-1)
    if blank_penalty > 0.0:
        log_probs = log_probs.clone()
        log_probs[:, :, blank_idx] -= blank_penalty

    log_probs_np = log_probs.cpu().numpy()
    B, max_T, V = log_probs_np.shape
    decoded_batch = []

    space_token = 27  # Default English space token

    for b in range(B):
        T = int(input_lengths[b].item()) if input_lengths is not None else max_T

        # Beam state: prefix_tuple -> (log_p_blank, log_p_non_blank)
        beam = {(): (0.0, -float("inf"))}

        for t in range(T):
            new_beam = defaultdict(lambda: (-float("inf"), -float("inf")))
            frame_probs = log_probs_np[b, t]

            for prefix, (p_b, p_nb) in beam.items():
                p_tot = np.logaddexp(p_b, p_nb)

                # 1. Blank extension: does not change prefix
                b_prob = frame_probs[blank_idx]
                curr_p_b, curr_p_nb = new_beam[prefix]
                new_beam[prefix] = (np.logaddexp(curr_p_b, p_tot + b_prob), curr_p_nb)

                # 2. Non-blank extensions
                for c in range(V):
                    if c == blank_idx:
                        continue
                    c_prob = frame_probs[c]
                    end_t = prefix[-1] if len(prefix) > 0 else None

                    if c == end_t:
                        # Same character repeated
                        # From blank -> distinct character added
                        new_pref = prefix + (c,)
                        curr_b, curr_nb = new_beam[new_pref]
                        new_beam[new_pref] = (curr_b, np.logaddexp(curr_nb, p_b + c_prob))

                        # From non-blank -> collapsed into existing character
                        curr_b, curr_nb = new_beam[prefix]
                        new_beam[prefix] = (curr_b, np.logaddexp(curr_nb, p_nb + c_prob))
                    else:
                        new_pref = prefix + (c,)
                        curr_b, curr_nb = new_beam[new_pref]
                        new_beam[new_pref] = (curr_b, np.logaddexp(curr_nb, p_tot + c_prob))

            # Score and prune beam to beam_width
            scored_beam = []
            for prefix, (p_b, p_nb) in new_beam.items():
                p_acoustic = np.logaddexp(p_b, p_nb)
                score = p_acoustic

                # Apply Language Model score if configured
                if lm is not None and vocab_map is not None and len(prefix) > 0:
                    text = "".join(vocab_map.get(tok, "") for tok in prefix)
                    words = text.strip().split()
                    if len(words) > 0:
                        lm_score = lm.score_sentence(text)
                        score += lm_weight * lm_score

                        # Bonus for matching valid lexicon words
                        if lexicon is not None:
                            matched_words = sum(1 for w in words if w in lexicon)
                            score += word_bonus * matched_words

                scored_beam.append((score, prefix, (p_b, p_nb)))

            scored_beam.sort(key=lambda x: x[0], reverse=True)
            beam = {pref: probs for _, pref, probs in scored_beam[:beam_width]}

        # Pick best prefix from top beam
        best_prefix = max(beam.keys(), key=lambda p: np.logaddexp(beam[p][0], beam[p][1]))
        decoded_batch.append(list(best_prefix))

    return decoded_batch
