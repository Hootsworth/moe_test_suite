"""
Unit Tests for Foundation Model Clinical MoE Adapter Architecture
"""

import unittest
import torch
from moe_framework.foundation_adapter import FoundationClinicalMoEAdapter


class TestFoundationAdapter(unittest.TestCase):

    def test_parameter_counts_and_freezing(self):
        """Verify backbone parameters are frozen and adapter is lightweight (<0.5%)."""
        model = FoundationClinicalMoEAdapter(
            adapter_dim=64,
            n_vocab=27,
            n_routed_experts=4,
            n_shared_experts=1,
            routing_mode="calibrated_atypical",
            freeze_backbone=True,
        )
        counts = model.count_parameters()
        self.assertGreater(counts["total_backbone"], 90_000_000)
        self.assertLess(counts["trainable_adapter"], 500_000)
        self.assertLess(counts["trainable_pct"], 0.5)  # Less than 0.5% trainable

    def test_forward_all_routing_modes(self):
        """Verify forward pass output shapes across all 4 benchmark configurations."""
        modes = ["linear_probe", "standard_single_gate", "decoupled_severity", "calibrated_atypical"]
        B, T, D = 2, 40, 768
        mock_features = torch.randn(B, T, D)
        lengths = torch.tensor([40, 32], dtype=torch.long)

        for mode in modes:
            model = FoundationClinicalMoEAdapter(
                adapter_dim=64,
                n_vocab=27,
                n_routed_experts=4,
                n_shared_experts=1,
                routing_mode=mode,
                freeze_backbone=True,
            )
            out = model(mock_features, lengths, severity="moderate_dysarthria")
            self.assertIn("logits", out)
            self.assertIn("log_probs_ctc", out)
            self.assertEqual(out["logits"].shape, (B, T, 28))
            self.assertEqual(out["log_probs_ctc"].shape, (T, B, 28))
            self.assertFalse(torch.isnan(out["log_probs_ctc"]).any())

    def test_acoustic_calibration_effect(self):
        """Verify that speaker acoustic-average changes routing logits in calibrated mode."""
        model = FoundationClinicalMoEAdapter(
            adapter_dim=64,
            n_vocab=27,
            n_routed_experts=4,
            n_shared_experts=1,
            routing_mode="calibrated_atypical",
            freeze_backbone=True,
        )
        # Two mock utterances with identical local frames but different global acoustic shifts
        B, T, D = 2, 30, 768
        mock_features = torch.randn(1, T, D).expand(B, -1, -1).clone()
        # Add acoustic offset to second utterance (simulating VTLN / vocal tract distortion)
        mock_features[1] = mock_features[1] + 2.5
        lengths = torch.tensor([30, 30], dtype=torch.long)

        out = model(mock_features, lengths, severity="severe_dysarthria")
        g_probs = out["G"].reshape(B, T, -1)
        # Gate probabilities should differ between utterance 0 and 1 due to acoustic calibration
        diff = (g_probs[0] - g_probs[1]).abs().mean().item()
        self.assertGreater(diff, 1e-4)


if __name__ == "__main__":
    unittest.main()
