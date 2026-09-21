"""
Unit Tests for TORGO Clinical Speech Pipeline & Disjoint Partitioning
"""

import unittest
import torch
from moe_framework.torgo_pipeline import (
    TORGO_SPEAKER_REGISTRY,
    TorgoPartitionManager,
    TorgoDataset,
    torgo_collate_fn,
)
from moe_framework.atypical_moe import AtypicalClinicalMoE
from moe_framework.production_speech_pipeline import EnglishPhoneticTokenizer


class TestTorgoPipeline(unittest.TestCase):

    def test_speaker_registry(self):
        """Verify all 15 TORGO speakers are catalogued with valid severity levels."""
        self.assertEqual(len(TORGO_SPEAKER_REGISTRY), 15)
        valid_severities = {
            "control_typical",
            "mild_dysarthria",
            "moderate_dysarthria",
            "severe_dysarthria",
        }
        for spk_id, meta in TORGO_SPEAKER_REGISTRY.items():
            self.assertIn(meta.severity, valid_severities)
            self.assertIn(meta.gender, {"M", "F"})
            self.assertIn(meta.etiology, {"Control", "Cerebral_Palsy", "ALS"})

    def test_canonical_disjoint_split(self):
        """Verify strict speaker-disjoint partition: S_train ∩ S_test = ∅."""
        train_spks, test_spks = TorgoPartitionManager.get_canonical_disjoint_split()
        train_set = set(train_spks)
        test_set = set(test_spks)

        # Zero speaker overlap
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(train_set) + len(test_set), 15)

        # Both train and test cohorts must cover all 4 severity tiers
        train_sevs = {TORGO_SPEAKER_REGISTRY[s].severity for s in train_spks}
        test_sevs = {TORGO_SPEAKER_REGISTRY[s].severity for s in test_spks}
        self.assertEqual(len(train_sevs), 4)
        self.assertEqual(len(test_sevs), 4)

    def test_dataset_and_collation(self):
        """Test dataset tensor dimensions and batch collation."""
        test_spks = ["MC04", "M05", "F03", "F01"]
        prompts = ["STICK", "WHEN HE SPEAKS HIS VOICE CRACKS"]
        utts = TorgoPartitionManager.synthesize_calibrated_torgo_cohort(test_spks, prompts, seed=123)
        dataset = TorgoDataset(utts)
        self.assertEqual(len(dataset), len(test_spks) * len(prompts))

        sample = dataset[0]
        self.assertEqual(sample["mel"].ndim, 2)
        self.assertEqual(sample["mel"].shape[1], 80)
        self.assertTrue(sample["target"].numel() > 0)

        # Collation
        batch = [dataset[i] for i in range(4)]
        collated = torgo_collate_fn(batch)
        self.assertEqual(collated["inputs"].shape[0], 4)
        self.assertEqual(collated["inputs"].shape[2], 80)
        self.assertEqual(collated["input_lengths"].shape[0], 4)
        self.assertEqual(collated["targets"].shape[0], 4)
        self.assertEqual(collated["target_lengths"].shape[0], 4)

    def test_moe_forward_on_torgo_batch(self):
        """Test forward pass of AtypicalClinicalMoE on collated TORGO batches."""
        tokenizer = EnglishPhoneticTokenizer()
        model = AtypicalClinicalMoE(
            in_dim=80,
            n_vocab=tokenizer.vocab_size - 1,
            routing_mode="calibrated_atypical",
        )
        model.eval()

        test_spks = ["MC04", "M05"]
        utts = TorgoPartitionManager.synthesize_calibrated_torgo_cohort(test_spks, ["TROUBLE"], seed=42)
        dataset = TorgoDataset(utts)
        batch = [dataset[0], dataset[1]]
        collated = torgo_collate_fn(batch)

        with torch.no_grad():
            out = model(collated["inputs"], collated["input_lengths"], severity="mild_dysarthria")
            self.assertIn("logits", out)
            self.assertIn("Gm", out)
            B, T_sub, V = out["logits"].shape
            self.assertEqual(B, 2)
            self.assertEqual(V, tokenizer.vocab_size)


if __name__ == "__main__":
    unittest.main()
