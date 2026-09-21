"""
Clinical Speech Pipeline for Atypical and Dysarthric Speech Recognition
======================================================================
Models multi-subsystem neuromotor speech pathologies across severity strata:
  - Control / Typical Speech (Healthy canonical reference)
  - Mild Dysarthria (Mild consonant imprecision, slight duration elongation)
  - Moderate Dysarthria (Vowel space centralization, temporal elongation x1.6)
  - Severe Dysarthria (Extreme centralization toward schwa, severe sluggishness x2.2, breathiness)

Strict Speaker-Disjoint Evaluation:
  Training and evaluation sets draw from mutually exclusive patient cohorts (S_train ∩ S_test = ∅).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from moe_framework.production_speech_pipeline import EnglishPhoneticTokenizer

SEVERITY_LEVELS = [
    "control_typical",
    "mild_dysarthria",
    "moderate_dysarthria",
    "severe_dysarthria",
]


class ClinicalPatientProfile:
    """
    Defines an individual patient's anatomical and neuromotor pathology profile.
    Captures patient-specific vocal tract length, vowel centralization factor,
    articulatory sluggishness, and glottal breathiness.
    """
    def __init__(self, patient_id: str, severity: str, rng: np.random.Generator):
        self.patient_id = patient_id
        self.severity = severity

        if severity == "control_typical":
            self.vocal_tract_scale = float(rng.uniform(0.98, 1.02))
            self.centralization_factor = 0.0  # Full canonical vowel space
            self.sluggishness_factor = float(rng.uniform(1.0, 1.1))  # Normal tempo
            self.glottal_noise_std = 0.2
            self.spectral_tilt = float(rng.uniform(0.0, 0.1))
        elif severity == "mild_dysarthria":
            self.vocal_tract_scale = float(rng.uniform(0.95, 1.05))
            self.centralization_factor = float(rng.uniform(0.15, 0.30))  # Slight undershoot
            self.sluggishness_factor = float(rng.uniform(1.2, 1.4))  # 20-40% slower
            self.glottal_noise_std = 0.35
            self.spectral_tilt = float(rng.uniform(0.1, 0.3))
        elif severity == "moderate_dysarthria":
            self.vocal_tract_scale = float(rng.uniform(0.92, 1.08))
            self.centralization_factor = float(rng.uniform(0.40, 0.60))  # Pronounced centralization
            self.sluggishness_factor = float(rng.uniform(1.5, 1.8))  # 50-80% slower
            self.glottal_noise_std = 0.55
            self.spectral_tilt = float(rng.uniform(0.3, 0.5))
        elif severity == "severe_dysarthria":
            self.vocal_tract_scale = float(rng.uniform(0.90, 1.10))
            self.centralization_factor = float(rng.uniform(0.70, 0.85))  # Collapse toward schwa
            self.sluggishness_factor = float(rng.uniform(2.0, 2.5))  # 2x slower + prolonged pauses
            self.glottal_noise_std = 0.80  # Severe breathiness / low SNR
            self.spectral_tilt = float(rng.uniform(0.5, 0.8))
        else:
            raise ValueError(f"Unknown severity level: {severity}")


class ClinicalSpeechWorld:
    """
    Synthesizes and manages clinical speech recordings across multiple patient cohorts
    with strict speaker-disjoint cross-validation partitions.
    """

    TRAIN_VOCABULARY = [
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
        "CLINICAL ASSESSMENT MEASURES MOTOR SPEECH INTELLIGIBILITY",
        "SPARSE MIXTURE OF EXPERTS COMPENSATES FOR ARTICULATORY DEFICITS",
        "DYSARTHRIC PATIENTS EXHIBIT CENTRALIZED VOWEL FORMANTS",
        "DECOUPLED ROUTING PREVENTS NEGATIVE CROSS ETIOLOGY TRANSFER",
        "INDEPENDENT GATING ISOLATES CONFLICTING GRADIENTS",
        "ACOUSTIC AVERAGES CALIBRATE PATIENT SPECIFIC BIAS TERMS",
        "TEMPORAL PROLONGATION REFLECTS MOTOR SLUGGISHNESS",
        "SHARED BACKBONES PRESERVE UNIVERSAL PHONETIC BOUNDARIES",
        "AUTOMATIC SPEECH RECOGNITION RESTORES COMMUNICATION INDEPENDENCE",
        "SPEECH THERAPY TARGETS RESPIRATORY AND LARYNGEAL CONTROL",
        "NEUROMOTOR DISORDERS ALTER VOCAL TRACT RESIDUALS",
    ]

    HELD_OUT_TEST_VOCABULARY = [
        "UNSEEN PATIENT EVALUATION DEMONSTRATES ZERO SHOT ADAPTATION",
        "CROSS SPEAKER GENERALIZATION OVERCOMES DATA SCARCITY",
        "CALIBRATED ROUTING NORMALIZES INDIVIDUAL ARTICULATORY DRIFT",
        "ROBUST CTC DECODING PREVENTS FALSE DELETION ERRORS",
        "PATHOLOGY COMPENSATORS EXPAND RESTRICTED ACOUSTIC ENVELOPES",
        "SPEAKER DISJOINT BENCHMARKING GUARANTEES UNBIASED CLINICAL VALIDATION",
    ]

    def __init__(self, seed: int = 42, n_train_patients_per_sev: int = 4, n_test_patients_per_sev: int = 2):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tokenizer = EnglishPhoneticTokenizer()
        self.n_mels = 80

        # Generate distinct patient cohorts (S_train ∩ S_test = ∅)
        self.train_patients: Dict[str, List[ClinicalPatientProfile]] = {}
        self.test_patients: Dict[str, List[ClinicalPatientProfile]] = {}

        p_idx = 1
        for sev in SEVERITY_LEVELS:
            self.train_patients[sev] = []
            for _ in range(n_train_patients_per_sev):
                p = ClinicalPatientProfile(f"PATIENT_TR_{p_idx:03d}", sev, self.rng)
                self.train_patients[sev].append(p)
                p_idx += 1

            self.test_patients[sev] = []
            for _ in range(n_test_patients_per_sev):
                p = ClinicalPatientProfile(f"PATIENT_TE_{p_idx:03d}", sev, self.rng)
                self.test_patients[sev].append(p)
                p_idx += 1

    def synthesize_clinical_utterance(
        self,
        patient: ClinicalPatientProfile,
        split: str = "train",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Synthesizes an 80-dim Log-Mel filterbank utterance for a specific clinical patient:
          - patient-specific formant scaling and centralization
          - patient-specific temporal elongation and pause jitter
          - returns (mel, target_tokens, utterance_mean, patient_id)
        """
        rng = rng or self.rng
        sentences = self.TRAIN_VOCABULARY if split == "train" else self.HELD_OUT_TEST_VOCABULARY
        sentence = rng.choice(sentences)

        words = sentence.split()
        max_words = min(len(words), rng.integers(3, 6))
        start_w = rng.integers(0, max(1, len(words) - max_words + 1))
        phrase = " ".join(words[start_w:start_w + max_words])

        token_ids = self.tokenizer.encode(phrase)
        if len(token_ids) == 0:
            token_ids = [1, 2, 3, 4]

        # Base frames per character scaled by patient sluggishness
        base_frames = rng.integers(2, 5)
        frames_per_char = max(2, int(round(base_frames * patient.sluggishness_factor)))
        n_chars = len(token_ids)
        total_frames = max(n_chars + 2, n_chars * frames_per_char)

        spec = rng.normal(0, 0.3 + patient.glottal_noise_std * 0.3, size=(total_frames, self.n_mels)).astype(np.float32)

        # Spectral tilt (high-frequency attenuation)
        tilt_weights = np.linspace(1.0, 1.0 - patient.spectral_tilt * 0.5, self.n_mels, dtype=np.float32)
        spec = spec * tilt_weights

        # Schwa center for dysarthric vowel centralization
        schwa_mel = 35  # ~1000-1500 Hz region

        for i, tok in enumerate(token_ids):
            start = i * frames_per_char
            end = min(total_frames, (i + 1) * frames_per_char)

            # Canonical formant center
            canonical_center = int(((tok * 3) % 65) + 8)

            # Apply vocal tract scale & centralization:
            # f_center = (1 - centralization) * canonical + centralization * schwa
            f_center = (1.0 - patient.centralization_factor) * (canonical_center * patient.vocal_tract_scale) \
                       + patient.centralization_factor * schwa_mel
            f_center = int(np.clip(f_center, 4, self.n_mels - 5))

            f_width = max(3, int(4 + patient.centralization_factor * 3))
            band_l = max(0, f_center - f_width)
            band_r = min(self.n_mels, f_center + f_width)
            spec[start:end, band_l:band_r] += float(2.2 * (1.0 - patient.glottal_noise_std * 0.3))

        mel_tensor = torch.from_numpy(spec)
        mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)

        # Utterance Acoustic Average (Long-Term Average Spectrum - LTAS): (80,)
        utterance_mean = mel_tensor.mean(dim=0)
        target_tensor = torch.tensor(token_ids, dtype=torch.long)

        return mel_tensor, target_tensor, utterance_mean, patient.patient_id


class ClinicalSpeechDataset(Dataset):
    """PyTorch Dataset for Clinical Speech."""
    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]], severity: str):
        self.samples = samples
        self.severity = severity

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        mel, targets, u_mean, p_id = self.samples[idx]
        return {
            "x": mel,
            "targets": targets,
            "utterance_mean": u_mean,
            "patient_id": p_id,
            "severity": self.severity,
        }


def clinical_speech_collate_fn(batch: List[Dict[str, Union[torch.Tensor, str]]]) -> Dict[str, Union[torch.Tensor, str]]:
    batch_size = len(batch)
    severity = batch[0]["severity"]

    in_lens = torch.tensor([item["x"].size(0) for item in batch], dtype=torch.long)
    tar_lens = torch.tensor([item["targets"].size(0) for item in batch], dtype=torch.long)

    max_T = int(in_lens.max().item())
    max_U = int(tar_lens.max().item())
    n_mels = batch[0]["x"].size(1)

    padded_x = torch.zeros(batch_size, max_T, n_mels, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_U, dtype=torch.long)
    u_means = torch.stack([item["utterance_mean"] for item in batch], dim=0)
    p_ids = [item["patient_id"] for item in batch]

    for i, item in enumerate(batch):
        t_l = in_lens[i]
        u_l = tar_lens[i]
        padded_x[i, :t_l] = item["x"]
        padded_targets[i, :u_l] = item["targets"]

    return {
        "x": padded_x,
        "targets": padded_targets,
        "input_lengths": in_lens,
        "target_lengths": tar_lens,
        "utterance_mean": u_means,
        "patient_ids": p_ids,
        "severity": severity,
    }


def create_clinical_dataloaders(
    world: ClinicalSpeechWorld,
    batch_size: int = 20,
    n_train_per_patient: int = 50,
    n_test_per_patient: int = 30,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """Creates speaker-disjoint train and test DataLoaders for all clinical severity strata."""
    train_loaders = {}
    test_loaders = {}

    rng_tr = np.random.default_rng(world.seed + 101)
    rng_te = np.random.default_rng(world.seed + 909)

    for sev in SEVERITY_LEVELS:
        tr_samples = []
        for p in world.train_patients[sev]:
            for _ in range(n_train_per_patient):
                tr_samples.append(world.synthesize_clinical_utterance(p, split="train", rng=rng_tr))

        te_samples = []
        for p in world.test_patients[sev]:
            for _ in range(n_test_per_patient):
                te_samples.append(world.synthesize_clinical_utterance(p, split="test", rng=rng_te))

        ds_tr = ClinicalSpeechDataset(tr_samples, sev)
        ds_te = ClinicalSpeechDataset(te_samples, sev)

        train_loaders[sev] = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=clinical_speech_collate_fn)
        test_loaders[sev] = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=clinical_speech_collate_fn)

    return train_loaders, test_loaders
