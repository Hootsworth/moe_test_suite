import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix of size d x d."""
    m = rng.normal(0, 1, size=(d, d))
    q, r = np.linalg.qr(m)
    q *= np.sign(np.diag(r))
    return q


class SyntheticMoEWorld:
    """
    Synthetic multi-task acoustic world representing phoneme recognition under
    multi-domain shifts:
      - Block A ("phone identity"): corrupted by domain-specific rotation C_task.
        - Language: strong rotation (phoneme inventory shift)
        - Child / Atypical: shared mild rotation
      - Block B ("prosody / deviation"): corrupted by additive warp vector w.
        - Language: no warp (0.0)
        - Child / Atypical: shared warp direction with task-specific magnitude.
      - Acoustic Continuous Signature z_acoustic:
        A continuous vector encoding acoustic metadata (pitch/vocal tract proxy + warp magnitude)
        for testing continuous acoustic conditioning without discrete task IDs.
    """

    def __init__(self, cfg: Dict[str, Union[int, float]]):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"])
        K, dA, dB = int(cfg["n_classes"]), int(cfg["dim_a"]), int(cfg["dim_b"])

        # Base class prototypes in representation space
        self.base_A = self.rng.normal(0, 1, size=(K, dA)) * float(cfg["class_scale"])
        self.base_B = self.rng.normal(0, 1, size=(K, dB)) * float(cfg["class_scale"])

        # Rotation matrices for Block A
        I_A = np.eye(dA)
        Q_mild = random_orthogonal(dA, self.rng)
        Q_strong = random_orthogonal(dA, self.rng)
        alpha_mild = float(cfg["alpha_mild"])
        alpha_strong = float(cfg["alpha_strong"])

        C_mild = (1 - alpha_mild) * I_A + alpha_mild * Q_mild
        C_strong = (1 - alpha_strong) * I_A + alpha_strong * Q_strong

        # Warp direction for Block B
        w = self.rng.normal(0, 1, size=(dB,))
        w /= np.linalg.norm(w)
        self.w = w

        self.task_C = {"language": C_strong, "child": C_mild, "atypical": C_mild}
        self.task_mag = {
            "language": float(cfg["mag_language"]),
            "child": float(cfg["mag_child"]),
            "atypical": float(cfg["mag_atypical"]),
        }

        # Continuous acoustic signatures (e.g. 4-dim acoustic embedding per task)
        # Dimensions: [warp_magnitude, rotation_magnitude, acoustic_pitch_shift, vocal_tract_ratio]
        self.task_acoustic_emb = {
            "language": np.array([0.0, alpha_strong, 0.0, 1.0], dtype=np.float32),
            "child": np.array([float(cfg["mag_child"]), alpha_mild, 1.5, 0.75], dtype=np.float32),
            "atypical": np.array([float(cfg["mag_atypical"]), alpha_mild, -0.5, 1.1], dtype=np.float32),
        }

    def sample_task(
        self,
        task: str,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample n data points for a given task.
        Returns:
            X: FloatTensor of shape (n, dim_a + dim_b)
            y: LongTensor of shape (n,)
            z_acoustic: FloatTensor of shape (n, acoustic_dim)
        """
        rng = rng or self.rng
        K = int(self.cfg["n_classes"])
        y = rng.integers(0, K, size=n)

        C = self.task_C[task]
        mag = self.task_mag[task]
        noise_std = float(self.cfg["noise_std"])

        A = self.base_A[y] @ C + rng.normal(0, noise_std, size=(n, int(self.cfg["dim_a"])))
        B = self.base_B[y] + mag * self.w[None, :] + rng.normal(0, noise_std, size=(n, int(self.cfg["dim_b"])))
        X_np = np.concatenate([A, B], axis=1).astype(np.float32)

        # Acoustic continuous embeddings with small per-utterance noise
        base_z = self.task_acoustic_emb[task]
        z_noise = rng.normal(0, 0.05, size=(n, len(base_z))).astype(np.float32)
        z_np = base_z[None, :] + z_noise

        X = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(y.astype(np.int64))
        z = torch.from_numpy(z_np)
        return X, y_tensor, z


class SyntheticMoEDataset(Dataset):
    """PyTorch Dataset wrapper for synthetic multi-task samples."""

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        z_acoustic: torch.Tensor,
        task_name: str,
    ):
        self.X = X
        self.y = y
        self.z_acoustic = z_acoustic
        self.task_name = task_name

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "z_acoustic": self.z_acoustic[idx],
            "task": self.task_name,
        }


def create_task_dataloaders(
    world: SyntheticMoEWorld,
    tasks: List[str],
    cfg: Dict[str, Union[int, float]],
    batch_size: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Creates PyTorch DataLoaders for each task.
    Returns:
        train_loaders: Dict[task_name -> DataLoader]
        val_loaders: Dict[task_name -> DataLoader]
    """
    train_loaders = {}
    val_loaders = {}

    rng_val = np.random.default_rng(int(cfg["seed"]) + 999)

    for task in tasks:
        X_tr, y_tr, z_tr = world.sample_task(task, int(cfg["n_train_per_task"]))
        X_va, y_va, z_va = world.sample_task(task, int(cfg["n_val_per_task"]), rng=rng_val)

        ds_tr = SyntheticMoEDataset(X_tr, y_tr, z_tr, task)
        ds_va = SyntheticMoEDataset(X_va, y_va, z_va, task)

        train_loaders[task] = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loaders[task] = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    return train_loaders, val_loaders
