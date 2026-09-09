"""
Continuous Speech Sequence Generator for Multi-Domain ASR Testing
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    m = rng.normal(0, 1, size=(d, d))
    q, r = np.linalg.qr(m)
    q *= np.sign(np.diag(r))
    return q


class SpeechSequenceWorld:
    """
    Generates multi-domain continuous speech utterances for CTC sequence modeling:
      - Class tokens y in {1 .. V} (0 is reserved for CTC blank).
      - Utterances are sequences of phoneme tokens (4-7 tokens per utterance).
      - Calibrated duration distributions:
        * Language: normal speech rate (2-4 frames/token, avg ~3.0 frames).
        * Child: higher variance (2-5 frames/token, avg ~3.5 frames).
        * Atypical: dysarthric elongation (4-7 frames/token, avg ~5.5 frames) + short pauses (1-2 frames).
    """

    def __init__(self, cfg: Dict[str, Union[int, float]]):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg["seed"]))
        self.n_vocab = int(cfg["n_vocab"])  # number of phoneme/character classes
        self.dA = int(cfg["dim_a"])
        self.dB = int(cfg["dim_b"])
        self.space_token = int(cfg.get("space_token", self.n_vocab + 1))
        self.total_vocab_size = self.n_vocab + 2  # 0: blank, 1..V: phonemes, V+1: space

        # Base phoneme embeddings in latent acoustic space
        self.base_A = self.rng.normal(0, 1, size=(self.total_vocab_size, self.dA)) * float(cfg["class_scale"])
        self.base_B = self.rng.normal(0, 1, size=(self.total_vocab_size, self.dB)) * float(cfg["class_scale"])

        # Rotations for Block A
        I_A = np.eye(self.dA)
        Q_mild = random_orthogonal(self.dA, self.rng)
        Q_strong = random_orthogonal(self.dA, self.rng)
        alpha_mild = float(cfg["alpha_mild"])
        alpha_strong = float(cfg["alpha_strong"])

        C_mild = (1 - alpha_mild) * I_A + alpha_mild * Q_mild
        C_strong = (1 - alpha_strong) * I_A + alpha_strong * Q_strong

        # Warp direction for Block B
        w = self.rng.normal(0, 1, size=(self.dB,))
        w /= np.linalg.norm(w)
        self.w = w

        self.task_C = {"language": C_strong, "child": C_mild, "atypical": C_mild}
        self.task_mag = {
            "language": float(cfg["mag_language"]),
            "child": float(cfg["mag_child"]),
            "atypical": float(cfg["mag_atypical"]),
        }

        # Continuous acoustic metadata vector
        self.task_acoustic_emb = {
            "language": np.array([0.0, alpha_strong, 0.0, 1.0], dtype=np.float32),
            "child": np.array([float(cfg["mag_child"]), alpha_mild, 1.5, 0.75], dtype=np.float32),
            "atypical": np.array([float(cfg["mag_atypical"]), alpha_mild, -0.5, 1.1], dtype=np.float32),
        }

    def generate_utterance(
        self,
        task: str,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single speech utterance:
        Returns:
            X: Acoustic frames (T_raw, dA + dB)
            targets: Target token sequence (U,) with values in {1 .. V}
            z_acoustic: Acoustic metadata vector (4,)
        """
        rng = rng or self.rng
        C = self.task_C[task]
        mag = self.task_mag[task]
        noise_std = float(self.cfg["noise_std"])

        # Target token sequence (4 to 7 tokens)
        seq_len = rng.integers(4, 8)
        target_tokens = rng.integers(1, self.n_vocab + 1, size=seq_len)

        # Realize acoustic frames for each target token
        frames_A, frames_B = [], []

        for token in target_tokens:
            # Calibrated duration ranges
            if task == "atypical":
                duration = rng.integers(4, 8)  # 4 to 7 frames
            elif task == "child":
                duration = rng.integers(2, 6)  # 2 to 5 frames
            else:
                duration = rng.integers(2, 5)  # 2 to 4 frames

            drift = rng.normal(0, 0.08, size=(duration, self.dA))
            base_proto_A = self.base_A[token:token + 1] @ C
            fA = np.repeat(base_proto_A, duration, axis=0) + drift + rng.normal(0, noise_std, size=(duration, self.dA))

            base_proto_B = self.base_B[token:token + 1] + mag * self.w[None, :]
            fB = np.repeat(base_proto_B, duration, axis=0) + rng.normal(0, noise_std, size=(duration, self.dB))

            frames_A.append(fA)
            frames_B.append(fB)

        # In atypical speech, short occasional pause frames (1-2 frames)
        if task == "atypical" and rng.random() > 0.6:
            pause_dur = rng.integers(1, 3)
            frames_A.append(rng.normal(0, noise_std * 0.5, size=(pause_dur, self.dA)))
            frames_B.append(rng.normal(0, noise_std * 0.5, size=(pause_dur, self.dB)))

        all_A = np.concatenate(frames_A, axis=0)
        all_B = np.concatenate(frames_B, axis=0)
        X = np.concatenate([all_A, all_B], axis=1).astype(np.float32)

        base_z = self.task_acoustic_emb[task]
        z_noise = rng.normal(0, 0.05, size=base_z.shape).astype(np.float32)
        z = base_z + z_noise

        return X, target_tokens.astype(np.int64), z


class SpeechSequenceDataset(Dataset):
    """PyTorch Dataset for multi-domain speech utterances."""

    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], task: str):
        self.samples = samples
        self.task = task

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        X_np, y_np, z_np = self.samples[idx]
        return {
            "x": torch.from_numpy(X_np),
            "targets": torch.from_numpy(y_np),
            "z_acoustic": torch.from_numpy(z_np),
            "task": self.task,
        }


def speech_sequence_collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, str]]]
) -> Dict[str, Union[torch.Tensor, str]]:
    """
    Collate function to pad variable-length acoustic frame sequences and target sequences.
    """
    batch_size = len(batch)
    task = batch[0]["task"]

    input_lengths = torch.tensor([item["x"].size(0) for item in batch], dtype=torch.long)
    target_lengths = torch.tensor([item["targets"].size(0) for item in batch], dtype=torch.long)

    max_T = int(input_lengths.max().item())
    max_U = int(target_lengths.max().item())
    feat_dim = batch[0]["x"].size(1)

    padded_x = torch.zeros(batch_size, max_T, feat_dim, dtype=torch.float32)
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
        "task": task,
    }


def create_sequence_dataloaders(
    world: SpeechSequenceWorld,
    tasks: List[str],
    cfg: Dict[str, Union[int, float]],
    batch_size: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """Create train and validation DataLoaders for all speech sequence tasks."""
    train_loaders = {}
    val_loaders = {}

    rng_tr = np.random.default_rng(int(cfg["seed"]) + 10)
    rng_va = np.random.default_rng(int(cfg["seed"]) + 9999)

    n_train = int(cfg.get("n_train_seq_per_task", 400))
    n_val = int(cfg.get("n_val_seq_per_task", 150))

    for task in tasks:
        train_samples = [world.generate_utterance(task, rng=rng_tr) for _ in range(n_train)]
        val_samples = [world.generate_utterance(task, rng=rng_va) for _ in range(n_val)]

        ds_tr = SpeechSequenceDataset(train_samples, task)
        ds_va = SpeechSequenceDataset(val_samples, task)

        train_loaders[task] = DataLoader(
            ds_tr, batch_size=batch_size, shuffle=True, collate_fn=speech_sequence_collate_fn
        )
        val_loaders[task] = DataLoader(
            ds_va, batch_size=batch_size, shuffle=False, collate_fn=speech_sequence_collate_fn
        )

    return train_loaders, val_loaders
