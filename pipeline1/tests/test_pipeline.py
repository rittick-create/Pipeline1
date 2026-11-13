"""Run with python -m unittest discover -s tests -v from pipeline1."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Data.create_datasets import SymileMIMICDataset
from Data.data_modules import PipelineDataModule
from Modules.cl_lit import clinical_metrics
from Models.prediction import StudentClassifier
from Modules.pacx_lit import CrossModalDistillation

ROOT = Path(__file__).resolve().parents[2] / "symile-mimic/symile-mimic-aws"
torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_early_stages_reject_test_before_any_read(self):
        for stage in ("labs", "distillation"):
            with patch("numpy.load", side_effect=AssertionError("Data opened")):
                with self.assertRaises(ValueError):
                    SymileMIMICDataset(ROOT, "test", stage)
            with self.assertRaises(ValueError):
                PipelineDataModule(stage, ROOT).setup("test")

    def test_stage3_uses_no_cxr_and_masks_unknowns(self):
        original = np.load

        def load(path, *args, **kwargs):
            if "cxr_" in str(path):
                raise AssertionError("CXR opened")
            return original(path, *args, **kwargs)

        with patch("numpy.load", side_effect=load):
            ds = SymileMIMICDataset(ROOT, "test", "classification")
            self.assertEqual(len(ds), 464)
            self.assertNotIn("cxr", ds[0])
            self.assertTrue((ds.labels[~ds.label_mask] == 0).all())
        p = torch.rand(3, 14)
        y = torch.ones(3, 14)
        mask = torch.zeros(3, 14, dtype=torch.bool)
        ap, roc = clinical_metrics(p, y, mask)
        self.assertTrue(ap.isnan().all() and roc.isnan().all())

    def test_freezing_and_gradient_boundaries(self):
        m = CrossModalDistillation()
        m.train()
        self.assertFalse(m.cxr_encoder.training)
        expected = list(
            m.ecg_encoder.backbone.stage_list[-1].block_list[-1].parameters()
        )
        actual = [p for p in m.ecg_encoder.parameters() if p.requires_grad]
        self.assertEqual({id(p) for p in actual}, {id(p) for p in expected})
        self.assertEqual(sum(p.numel() for p in actual), 3418624)
        self.assertTrue(all(p.requires_grad for p in m.labs_encoder.parameters()))
        probe = StudentClassifier()
        probe.train()
        self.assertFalse(probe.ecg_encoder.training)
        self.assertFalse(probe.labs_encoder.training)
        logits = probe(torch.randn(2, 12, 256), torch.rand(2, 50), torch.ones(2, 50))
        logits.square().mean().backward()
        self.assertEqual(logits.shape, (2, 14))
        self.assertIsNotNone(probe.head.weight.grad)
        self.assertTrue(all(p.grad is None for p in probe.ecg_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in probe.labs_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
