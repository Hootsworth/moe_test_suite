"""
Multi-Gate Mixture of Experts (MoE) PyTorch Framework
"""

from .models import MultiGateMoEClassifier, Expert, ExpertPool, TaskGate
from .losses import StandardLoadBalanceLoss, DomainConditionalLoadBalanceLoss, DynamicBiasTracker
from .data import SyntheticMoEWorld, SyntheticMoEDataset, create_task_dataloaders
from .metrics import (
    compute_gate_statistics,
    compute_gate_cosine_similarity,
    compute_gini_specialization,
    run_expert_ablation,
    evaluate_hypothesis,
)
from .ctc_engine import levenshtein_distance, ctc_greedy_decode, SequenceErrorEvaluator
from .sequence_data import SpeechSequenceWorld, SpeechSequenceDataset, create_sequence_dataloaders
from .sequence_models import SequenceMultiGateMoE, Conv1DSubsampling
from .audio_features import LogMelSpectrogramExtractor, SpecAugment, extract_acoustic_signature
from .real_data_pipeline import DevanagariTokenizer, VaaniSpeechWorld, RealSpeechDataset, create_vaani_dataloaders

__all__ = [
    "MultiGateMoEClassifier",
    "SequenceMultiGateMoE",
    "Conv1DSubsampling",
    "Expert",
    "ExpertPool",
    "TaskGate",
    "StandardLoadBalanceLoss",
    "DomainConditionalLoadBalanceLoss",
    "DynamicBiasTracker",
    "SyntheticMoEWorld",
    "SyntheticMoEDataset",
    "create_task_dataloaders",
    "SpeechSequenceWorld",
    "SpeechSequenceDataset",
    "create_sequence_dataloaders",
    "LogMelSpectrogramExtractor",
    "SpecAugment",
    "extract_acoustic_signature",
    "DevanagariTokenizer",
    "VaaniSpeechWorld",
    "RealSpeechDataset",
    "create_vaani_dataloaders",
    "levenshtein_distance",
    "ctc_greedy_decode",
    "SequenceErrorEvaluator",
    "compute_gate_statistics",
    "compute_gate_cosine_similarity",
    "compute_gini_specialization",
    "run_expert_ablation",
    "evaluate_hypothesis",
]
