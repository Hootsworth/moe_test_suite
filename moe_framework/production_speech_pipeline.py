"""
Open-Vocabulary Speech Production Pipeline for Adult, Child, and Dysarthric Speech
with Strictly Disjoint 3-Way Train/Val/Test Splits
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EnglishPhoneticTokenizer:
    """
    Character and Phoneme Tokenizer for Standard English Speech:
      - Index 0: CTC Blank token (<blank>)
      - Index 1..26: Alphabet characters 'A' through 'Z'
      - Index 27: Space token (' ')
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

        self.blank_id = 0
        self.id_to_char[0] = "<blank>"

        for idx, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=1):
            self.char_to_id[ch] = idx
            self.id_to_char[idx] = ch

        self.space_id = 27
        self.char_to_id[' '] = self.space_id
        self.id_to_char[self.space_id] = " "

        self.vocab_size = len(self.id_to_char)

    def encode(self, text: str) -> List[int]:
        tokens = []
        for ch in text.upper():
            if ch in self.char_to_id:
                tokens.append(self.char_to_id[ch])
        return tokens

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for t in tokens:
            if t == self.blank_id:
                continue
            chars.append(self.id_to_char.get(t, ""))
        return "".join(chars)


class ProductionSpeechWorld:
    """
    Models physical speech production characteristics across 3 distinct human vocal-tract conditions:
      1. 'adult_speech' (LibriSpeech / Standard Adult):
         - Vocal tract length ~17 cm (standard formant distribution across 80 mel bands).
         - Standard articulatory rate (2-4 frames per character).
      2. 'child_speech' (MyST / CSLU Kids):
         - Vocal tract length ~11-13 cm (elevated formants +30% shifted into higher mel bands).
         - Higher fundamental frequency F0 + developmental duration variance (2-5 frames).
      3. 'dysarthric_speech' (TORGO / UASpeech):
         - Motor-speech pathology (vowel space centralization, articulatory sluggishness).
         - Prolonged vowel/frame durations (4-7 frames per char) + articulatory drift.

    STRICT 3-WAY DISJOINT SPLITS:
      Train, Validation, and Test pools draw from mutually exclusive, non-overlapping sentence corpora.
    """

    TRAIN_SENTENCES = [
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "SCIENTIFIC DISCOVERY REQUIRES PATIENCE AND RIGOR",
        "EXPERTS ROUTE ACOUSTIC SIGNALS DYNAMICALLY",
        "SPEECH RECOGNITION REQUIRES ROBUST ACOUSTIC MODELS",
        "CHILD SPEECH DEVIATES IN VOCAL TRACT LENGTH",
        "MOTOR PATHOLOGIES CAUSE ARTICULATORY SLUGGISHNESS",
        "SPARSE MIXTURE OF EXPERTS SCALES EFFICIENTLY",
        "DEEP LEARNING HAS REVOLUTIONIZED NATURAL LANGUAGE",
        "CONTINUOUS ACOUSTIC EMBEDDINGS PRESERVE DYNAMICS",
        "DISJOINT VOCABULARY TESTING PREVENTS MEMORIZATION",
        "THE SUN RISES IN THE EAST AND SETS IN THE WEST",
        "EVERY STEP FORWARD ADVANCES SCIENTIFIC KNOWLEDGE",
        "A BEAUTIFUL MORNING IN THE COUNTRYSIDE",
        "TECHNOLOGY EMPOWERS INNOVATION AND PROGRESS",
        "THE UNIVERSITY CONDUCTS RESEARCH AND EDUCATION",
        "ROBUST EVALUATION REQUIRES STATISTICAL SWEEPS",
    ]

    VAL_SENTENCES = [
        "INDEPENDENT VALIDATION VERIFIES GENERALIZATION",
        "UNSEEN SENTENCES TEST TRUE ACOUSTIC DECODING",
        "ACOUSTIC FORMANT DYNAMICS VARY ACROSS AGES",
        "NEURAL NETWORKS ALIGN SEQUENCES WITH CTC LOSS",
        "SPEECH PRODUCTION INVOLVES COMPLEX MOTOR CONTROL",
        "VOICE CHARACTERISTICS DEPEND ON VOCAL ANATOMY",
        "PROSODIC WARPING ALTERS TEMPORAL TRAJECTORIES",
        "RELIABLE ASR DEMANDS MULTI DOMAIN ROBUSTNESS",
    ]

    TEST_SENTENCES = [
        "FINAL HELD OUT BENCHMARK EVALUATES PURE ZERO SHOT GENERALIZATION",
        "CROSS DOMAIN PHONETICS REQUIRE SPECIALIZED SUB NETWORKS",
        "RIGOROUS STATISTICAL POWER CONFIRMS HYPOTHESIS SIGNIFICANCE",
        "PHYSICAL ACOUSTIC PROPERTIES DICTATE ROUTER CONVERGENCE",
        "DECOUPLED GATES ISOLATE CONFLICTING MULTI TASK GRADIENTS",
        "STANDALONE TEST CORPORA GUARANTEE UNBIASED PERFORMANCE ESTIMATES",
        "SPARSE ROUTING PRESERVES SUB LINEAR COMPUTATIONAL COMPLEXITY",
        "INVARIANT BACKBONES ANCHOR UNIVERSAL PHONETIC TRANSITIONS",
    ]

    def __init__(self, cfg: Dict[str, Union[int, float]]):
        self.cfg = cfg
        self.seed = int(cfg.get("seed", 7))
        self.rng = np.random.default_rng(self.seed)
        self.tokenizer = EnglishPhoneticTokenizer()
        self.n_mels = 80

        # Verify strict 3-way disjointness
        s_tr = set(self.TRAIN_SENTENCES)
        s_va = set(self.VAL_SENTENCES)
        s_te = set(self.TEST_SENTENCES)
        assert len(s_tr.intersection(s_va)) == 0, "Train and Val must be disjoint!"
        assert len(s_tr.intersection(s_te)) == 0, "Train and Test must be disjoint!"
        assert len(s_va.intersection(s_te)) == 0, "Val and Test must be disjoint!"

        # Domain acoustic signatures (energy, spectral centroid, spectral tilt, dynamics)
        self.domain_acoustic_signatures = {
            "adult_speech": np.array([0.0, 0.4, 0.1, 0.3], dtype=np.float32),
            "child_speech": np.array([0.2, 0.7, -0.4, 0.6], dtype=np.float32),      # elevated centroid, high F0
            "dysarthric_speech": np.array([-0.1, 0.3, 0.5, 0.8], dtype=np.float32), # low centroid, high duration variance
        }

    def synthesize_utterance(
        self,
        domain: str,
        split: str = "train",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synthesize 80-dim Log-Mel filterbank sequence and target token transcript:
        Returns:
            mel: (T_frames, 80) float tensor
            targets: (U,) long tensor of English token IDs
            z_acoustic: (4,) float tensor of acoustic metadata
        """
        rng = rng or self.rng
        if split == "test":
            sentences = self.TEST_SENTENCES
        elif split == "val":
            sentences = self.VAL_SENTENCES
        else:
            sentences = self.TRAIN_SENTENCES

        full_sentence = rng.choice(sentences)

        # Extract a random sub-phrase (3 to 6 words) for natural duration diversity
        words = full_sentence.split()
        max_words = min(len(words), rng.integers(3, 6))
        start_w = rng.integers(0, max(1, len(words) - max_words + 1))
        phrase = " ".join(words[start_w:start_w + max_words])

        token_ids = self.tokenizer.encode(phrase)
        if len(token_ids) == 0:
            token_ids = [1, 2, 3, 4]

        # Duration per character
        if domain == "dysarthric_speech":
            frames_per_char = rng.integers(4, 8)  # 4 to 7 frames (sluggish articulation)
        elif domain == "child_speech":
            frames_per_char = rng.integers(2, 6)  # 2 to 5 frames (developmental jitter)
        else:
            frames_per_char = rng.integers(2, 5)  # 2 to 4 frames (standard adult rate)

        n_chars = len(token_ids)
        total_frames = max(n_chars + 2, n_chars * frames_per_char)

        # Base 80-mel spectrogram
        base_spec = rng.normal(0, 0.4, size=(total_frames, self.n_mels)).astype(np.float32)

        for i, tok in enumerate(token_ids):
            start = i * frames_per_char
            end = min(total_frames, (i + 1) * frames_per_char)

            # Formant band assignment per phoneme
            if domain == "child_speech":
                # Elevated formants (+25% shifted to higher mel bands due to shorter vocal tract)
                f_center = int(((tok * 3) % 60) * 1.25 + 10)
            elif domain == "dysarthric_speech":
                # Centralized vowel space (compressed formant range in middle mel bands)
                f_center = int(30 + ((tok * 2) % 25))
            else:
                # Standard adult formant spread
                f_center = int((tok * 3) % self.n_mels)

            f_width = 4
            band_low = max(0, f_center - f_width)
            band_high = min(self.n_mels, f_center + f_width)
            base_spec[start:end, band_low:band_high] += 2.0

        # Utterance normalization
        mel_tensor = torch.from_numpy(base_spec)
        mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)

        target_tensor = torch.tensor(token_ids, dtype=torch.long)
        z_base = self.domain_acoustic_signatures[domain]
        z_noise = rng.normal(0, 0.05, size=z_base.shape).astype(np.float32)
        z_tensor = torch.from_numpy(z_base + z_noise)

        return mel_tensor, target_tensor, z_tensor


class ProductionSpeechDataset(Dataset):
    """PyTorch Dataset for Open-Vocabulary Speech Production."""

    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], domain: str):
        self.samples = samples
        self.domain = domain

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        mel, targets, z = self.samples[idx]
        return {
            "x": mel,
            "targets": targets,
            "z_acoustic": z,
            "domain": self.domain,
        }


def production_speech_collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, str]]]
) -> Dict[str, Union[torch.Tensor, str]]:
    batch_size = len(batch)
    domain = batch[0]["domain"]

    input_lengths = torch.tensor([item["x"].size(0) for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item["targets"].size(0) for item in batch], dtype=torch.long)

    max_T = int(input_lengths.max().item())
    max_U = int(target_lengths.max().item())
    n_mels = batch[0]["x"].size(1)

    padded_x = torch.zeros(batch_size, max_T, n_mels, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_U, dtype=torch.long)
    z_acoustics = torch.stack([item["z_acoustic"] for item in batch], dim=0)

    for i, item in enumerate(batch):
        t_len = input_lengths[i]
        u_len = target_lengths[i]
        padded_x[i, :t_len] = item["x"]
        padded_targets[i, :u_len] = item["targets"]

    return {
        "x": padded_x,
        "targets": padded_targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "z_acoustic": z_acoustics,
        "domain": domain,
    }


def create_production_dataloaders(
    world: ProductionSpeechWorld,
    domains: List[str],
    cfg: Dict[str, Union[int, float]],
    batch_size: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], Dict[str, DataLoader]]:
    """Create train, validation, and test DataLoaders with strictly disjoint 3-way sentence pools."""
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    seed = int(cfg["seed"])
    rng_tr = np.random.default_rng(seed + 202)
    rng_va = np.random.default_rng(seed + 8888)
    rng_te = np.random.default_rng(seed + 99999)

    n_train = int(cfg.get("n_train_per_domain", 300))
    n_val = int(cfg.get("n_val_per_domain", 60))
    n_test = int(cfg.get("n_test_per_domain", 60))

    for domain in domains:
        train_samples = [world.synthesize_utterance(domain, split="train", rng=rng_tr) for _ in range(n_train)]
        val_samples = [world.synthesize_utterance(domain, split="val", rng=rng_va) for _ in range(n_val)]
        test_samples = [world.synthesize_utterance(domain, split="test", rng=rng_te) for _ in range(n_test)]

        ds_tr = ProductionSpeechDataset(train_samples, domain)
        ds_va = ProductionSpeechDataset(val_samples, domain)
        ds_te = ProductionSpeechDataset(test_samples, domain)

        train_loaders[domain] = DataLoader(
            ds_tr, batch_size=batch_size, shuffle=True, collate_fn=production_speech_collate_fn
        )
        val_loaders[domain] = DataLoader(
            ds_va, batch_size=batch_size, shuffle=False, collate_fn=production_speech_collate_fn
        )
        test_loaders[domain] = DataLoader(
            ds_te, batch_size=batch_size, shuffle=False, collate_fn=production_speech_collate_fn
        )

    return train_loaders, val_loaders, test_loaders
