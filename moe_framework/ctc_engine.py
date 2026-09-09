"""
CTC Decoding and Levenshtein Distance (WER / CER) Engine
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch


def levenshtein_distance(
    ref: List[Union[int, str]],
    hyp: List[Union[int, str]],
) -> Dict[str, int]:
    """
    Compute exact Levenshtein alignment between reference and hypothesis tokens.
    Returns:
        Dict with 'substitutions', 'deletions', 'insertions', 'hits', and 'total_ref'.
    """
    R = len(ref)
    H = len(hyp)

    # DP table for edit distance: dp[i][j] = (dist, S, D, I)
    dp = np.zeros((R + 1, H + 1), dtype=int)

    for i in range(R + 1):
        dp[i, 0] = i
    for j in range(H + 1):
        dp[0, j] = j

    for i in range(1, R + 1):
        for j in range(1, H + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                sub = dp[i - 1, j - 1] + 1
                delete = dp[i - 1, j] + 1
                insert = dp[i, j - 1] + 1
                dp[i, j] = min(sub, delete, insert)

    # Backtracking to extract exact S, D, I
    i, j = R, H
    subs, dels, inss, hits = 0, 0, 0, 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i, j] == dp[i, j - 1] + 1:
            inss += 1
            j -= 1
        else:
            if i > 0:
                dels += 1
                i -= 1
            elif j > 0:
                inss += 1
                j -= 1

    return {
        "substitutions": subs,
        "deletions": dels,
        "insertions": inss,
        "hits": hits,
        "total_ref": R,
        "distance": int(dp[R, H]),
    }


def ctc_greedy_decode(
    logits: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    blank_idx: int = 0,
) -> List[List[int]]:
    """
    Greedy CTC argmax decoding: collapses consecutive duplicate tokens and removes blanks.
    Args:
        logits: (B, T, V) or (T, B, V) logits tensor
        input_lengths: Optional (B,) tensor with valid sequence lengths per sample
        blank_idx: Index of CTC blank token (default 0)
    Returns:
        List of decoded integer token lists per batch item
    """
    if logits.ndim == 3 and logits.size(0) != input_lengths.size(0) if input_lengths is not None else False:
        # If shape is (T, B, V), transpose to (B, T, V)
        logits = logits.transpose(0, 1)

    preds = logits.argmax(dim=-1).cpu().numpy()  # (B, T)
    batch_size, max_t = preds.shape
    decoded_sequences = []

    for b in range(batch_size):
        length = int(input_lengths[b].item()) if input_lengths is not None else max_t
        seq = preds[b, :length]

        collapsed = []
        prev = None
        for token in seq:
            if token != prev:
                if token != blank_idx:
                    collapsed.append(int(token))
                prev = token

        decoded_sequences.append(collapsed)

    return decoded_sequences


class SequenceErrorEvaluator:
    """
    Tracks and computes Word Error Rate (WER) and Character Error Rate (CER)
    along with S/D/I error decomposition.
    """

    def __init__(self, vocab_map: Optional[Dict[int, str]] = None, space_token: Optional[int] = None):
        self.vocab_map = vocab_map or {i: chr(ord('A') + i - 1) for i in range(1, 27)}
        self.space_token = space_token
        self.reset()

    def reset(self):
        self.stats = {
            "char_substitutions": 0,
            "char_deletions": 0,
            "char_insertions": 0,
            "char_hits": 0,
            "char_total": 0,
            "word_substitutions": 0,
            "word_deletions": 0,
            "word_insertions": 0,
            "word_hits": 0,
            "word_total": 0,
        }

    def tokens_to_words(self, tokens: List[int]) -> List[str]:
        """Convert a sequence of token IDs to a list of word strings."""
        if not tokens:
            return []
        if self.space_token is not None:
            words = []
            cur_word = []
            for t in tokens:
                if t == self.space_token:
                    if cur_word:
                        words.append("".join(cur_word))
                        cur_word = []
                else:
                    char = self.vocab_map.get(t, f"<{t}>")
                    cur_word.append(char)
            if cur_word:
                words.append("".join(cur_word))
            return words
        else:
            # If no space token, each token is treated as an individual token word
            return [self.vocab_map.get(t, f"<{t}>") for t in tokens]

    def update(
        self,
        ref_tokens_list: List[List[int]],
        hyp_tokens_list: List[List[int]],
    ):
        """Update metrics given lists of reference and hypothesis token sequences."""
        for ref_tokens, hyp_tokens in zip(ref_tokens_list, hyp_tokens_list):
            # 1. CER (character-level / phoneme-level)
            c_res = levenshtein_distance(ref_tokens, hyp_tokens)
            self.stats["char_substitutions"] += c_res["substitutions"]
            self.stats["char_deletions"] += c_res["deletions"]
            self.stats["char_insertions"] += c_res["insertions"]
            self.stats["char_hits"] += c_res["hits"]
            self.stats["char_total"] += c_res["total_ref"]

            # 2. WER (word-level)
            ref_words = self.tokens_to_words(ref_tokens)
            hyp_words = self.tokens_to_words(hyp_tokens)
            w_res = levenshtein_distance(ref_words, hyp_words)
            self.stats["word_substitutions"] += w_res["substitutions"]
            self.stats["word_deletions"] += w_res["deletions"]
            self.stats["word_insertions"] += w_res["insertions"]
            self.stats["word_hits"] += w_res["hits"]
            self.stats["word_total"] += w_res["total_ref"]

    def compute(self) -> Dict[str, float]:
        """Compute aggregated WER, CER, and percentage error components."""
        # Character Error Rate
        c_tot = max(1, self.stats["char_total"])
        cer = (self.stats["char_substitutions"] + self.stats["char_deletions"] + self.stats["char_insertions"]) / c_tot

        # Word Error Rate
        w_tot = max(1, self.stats["word_total"])
        wer = (self.stats["word_substitutions"] + self.stats["word_deletions"] + self.stats["word_insertions"]) / w_tot

        return {
            "WER": round(float(wer * 100.0), 2),
            "CER": round(float(cer * 100.0), 2),
            "word_sub_rate": round(float(self.stats["word_substitutions"] / w_tot * 100.0), 2),
            "word_del_rate": round(float(self.stats["word_deletions"] / w_tot * 100.0), 2),
            "word_ins_rate": round(float(self.stats["word_insertions"] / w_tot * 100.0), 2),
            "char_sub_rate": round(float(self.stats["char_substitutions"] / c_tot * 100.0), 2),
            "char_del_rate": round(float(self.stats["char_deletions"] / c_tot * 100.0), 2),
            "char_ins_rate": round(float(self.stats["char_insertions"] / c_tot * 100.0), 2),
            "total_words": self.stats["word_total"],
            "total_chars": self.stats["char_total"],
        }
