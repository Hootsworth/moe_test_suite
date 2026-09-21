"""
Unit Tests for Unified Multilingual Clinical Speech Pipeline
"""

import unittest
from moe_framework.multilingual_clinical_pipeline import (
    UnifiedMultilingualCorpus,
    MultilingualClinicalTokenizer,
    SEVERITY_LEVELS,
)


class TestMultilingualPipeline(unittest.TestCase):

    def setUp(self):
        self.corpus = UnifiedMultilingualCorpus()
        self.tokenizer = MultilingualClinicalTokenizer()

    def test_corpus_indexing(self):
        """Verify all 4 languages are represented with valid participant counts."""
        summary = self.corpus.get_summary()
        self.assertGreaterEqual(summary["total_participants"], 350)
        self.assertGreaterEqual(summary["total_audio_files"], 2000)

        # Check languages
        by_lang = summary["by_language"]
        self.assertIn("en", by_lang)
        self.assertIn("es", by_lang)
        self.assertIn("de", by_lang)
        self.assertIn("it", by_lang)
        self.assertGreaterEqual(by_lang["en"], 15)
        self.assertGreaterEqual(by_lang["es"], 90)
        self.assertGreaterEqual(by_lang["de"], 170)
        self.assertGreaterEqual(by_lang["it"], 50)

    def test_severity_levels(self):
        """Verify participants are mapped into all 4 clinical severity tiers."""
        summary = self.corpus.get_summary()
        by_sev = summary["by_severity"]
        for s in SEVERITY_LEVELS:
            self.assertIn(s, by_sev)
            self.assertGreater(by_sev[s], 0)

    def test_speaker_disjoint_split(self):
        """Verify train and test participant sets have zero intersection."""
        tr, te = self.corpus.get_disjoint_multilingual_split(seed=42)
        overlap = set(tr).intersection(set(te))
        self.assertEqual(len(overlap), 0)
        self.assertEqual(len(tr) + len(te), len(self.corpus.participants))

    def test_tokenizer_multilingual_support(self):
        """Verify tokenizer handles English, Spanish, German, and Italian text."""
        samples = [
            "WHEN HE SPEAKS HIS VOICE QUIVERS",
            "EL PUENTE DE LOS TRES ARCOS EN ESPAÑOL",
            "NORDWIND UND SONNE ÜBER DIE STRAßE",
            "IL RAMARRO VERDE DELLA ZIA",
        ]
        for text in samples:
            encoded = self.tokenizer.encode(text)
            self.assertTrue(len(encoded) > 0)
            decoded = self.tokenizer.decode(encoded)
            # Verify letters are preserved
            self.assertTrue(any(c.isalpha() for c in decoded))


if __name__ == "__main__":
    unittest.main()
