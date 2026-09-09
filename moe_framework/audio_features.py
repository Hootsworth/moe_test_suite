"""
Acoustic Feature Extraction and SpecAugment Pipeline for Real Speech
"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Construct a triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)
    in pure PyTorch for universal platform compatibility.
    """
    f_max = f_max or (sr / 2.0)
    # Convert Hz to Mel
    mel_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + f_max / 700.0)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return torch.from_numpy(fb)


class LogMelSpectrogramExtractor(nn.Module):
    """
    Extracts 80-dimensional Log-Mel filterbank features from raw 16kHz audio waveforms:
      Input:  (B, num_samples) or (num_samples,) waveform tensor at 16kHz.
      Output: (B, T_frames, 80) normalized log-mel spectrogram.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,  # 25ms
        hop_length: int = 160,  # 10ms
        n_mels: int = 80,
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Register triangular filterbank and Hann window
        fb = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer("filterbank", fb)  # (80, 257)
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, num_samples) float tensor in [-1.0, 1.0]
        Returns:
            log_mel: (B, T, n_mels)
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # STFT: (B, 257, T, 2) -> Complex magnitude (B, 257, T)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        mag = torch.abs(stft)  # (B, 257, T)

        # Apply Mel Filterbank: (80, 257) @ (B, 257, T) -> (B, 80, T)
        mel = torch.matmul(self.filterbank, mag)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))  # (B, 80, T)

        # Transpose to (B, T, 80) and normalize per utterance
        log_mel = log_mel.transpose(1, 2)
        mean = log_mel.mean(dim=1, keepdim=True)
        std = log_mel.std(dim=1, keepdim=True) + 1e-6
        norm_log_mel = (log_mel - mean) / std

        return norm_log_mel


class SpecAugment(nn.Module):
    """
    SpecAugment data augmentation: applies random Frequency Masking and Time Masking.
    """

    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 25):
        super().__init__()
        self.F = freq_mask_param
        self.T = time_mask_param

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, T, n_mels)
        """
        if not self.training:
            return mel

        B, T, D = mel.shape
        mel = mel.clone()

        # Frequency Masking
        f_len = torch.randint(0, self.F, (B,), device=mel.device)
        f_start = torch.randint(0, max(1, D - self.F), (B,), device=mel.device)
        for b in range(B):
            mel[b, :, f_start[b]:f_start[b] + f_len[b]] = 0.0

        # Time Masking
        t_len = torch.randint(0, min(self.T, T // 4 + 1), (B,), device=mel.device)
        t_start = torch.randint(0, max(1, T - self.T), (B,), device=mel.device)
        for b in range(B):
            mel[b, t_start[b]:t_start[b] + t_len[b], :] = 0.0

        return mel


def extract_acoustic_signature(mel_spec: torch.Tensor) -> torch.Tensor:
    """
    Extract a 4-dimensional continuous acoustic metadata vector z_acoustic from
    normalized log-mel spectrogram:
      1. Energy Mean / RMS proxy
      2. Spectral Centroid proxy (high vs low frequency concentration)
      3. Spectral Tilt / Slope proxy
      4. Temporal Variance / Dynamics proxy
    Returns:
        z_acoustic: (B, 4) tensor
    """
    if mel_spec.ndim == 2:
        mel_spec = mel_spec.unsqueeze(0)

    B, T, D = mel_spec.shape
    # 1. Energy
    energy = mel_spec.mean(dim=(1, 2))  # (B,)

    # 2. Spectral Centroid proxy
    freq_weights = torch.linspace(0.1, 1.0, D, device=mel_spec.device).unsqueeze(0).unsqueeze(0)
    centroid = (mel_spec * freq_weights).mean(dim=(1, 2))  # (B,)

    # 3. Spectral Tilt: difference between low and high mel bins
    low_band = mel_spec[:, :, :D // 2].mean(dim=(1, 2))
    high_band = mel_spec[:, :, D // 2:].mean(dim=(1, 2))
    tilt = low_band - high_band  # (B,)

    # 4. Temporal Dynamics: variance over time
    dynamics = mel_spec.var(dim=1).mean(dim=1)  # (B,)

    z = torch.stack([energy, centroid, tilt, dynamics], dim=-1)
    return z
