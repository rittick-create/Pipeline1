"""Check the new adapter against upstream preprocessing and real stored inputs."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import ECGFounderEncoder, MainModel
from utils import SymileMIMICDataset, TEST_CONDITIONS
from utils.ecg_preprocessing import preprocess_ecg, LEAD_ORDER

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(2)


class ECGFounderTests(unittest.TestCase):
    def test_preprocessing_matches_upstream_on_stored_records(self):
        spec = importlib.util.spec_from_file_location(
            "reference_ecgfounder_util", ROOT / "ecgfounder/util.py")
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        data = np.load(ROOT / "symile-mimic/symile-mimic-aws/data_npy/train/ecg_train.npy",
                       mmap_mode="r")
        for index in (0, 37, 9999):
            signal = np.array(data[index, 0], dtype=np.float64).T[LEAD_ORDER]
            expected = upstream.filter_bandpass(signal, 500)
            expected = ((expected - expected.mean()) / (expected.std() + 1e-8)).astype(np.float32)
            actual = preprocess_ecg(data[index])
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.shape, (12, 5000))
            self.assertTrue(np.isfinite(actual).all())
        with self.assertRaises(ValueError):
            preprocess_ecg(np.zeros((1, 5000, 12), dtype=np.float32))

    def test_frozen_backbone_keeps_new_branches_trainable(self):
        model = ECGFounderEncoder(hidden_size=8, dropout=0, freeze_backbone=True)
        model.train()
        self.assertFalse(model.backbone.training)
        shared, distinct, pred = model(torch.randn(2, 12, 256))
        (shared.square().mean() + distinct.square().mean() + pred.mean()).backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertIsNotNone(model.shared[0].weight.grad)
        self.assertIsNotNone(model.distinct[0].weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertEqual(pred.shape, (2, 14))

    def test_real_full_length_sample_runs_all_three_conditions(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample = SymileMIMICDataset(ROOT / "symile-mimic/symile-mimic-aws", "train")[0]
        model = MainModel(hidden_size=8, ecg_dropout=0, labs_dropout=0).to(device).eval()
        with torch.inference_mode():
            for condition, names in TEST_CONDITIONS.items():
                output = model(**{name: sample[name].unsqueeze(0).to(device) for name in names})
                self.assertEqual(output["pred_final"].shape, (1, 14), condition)
                self.assertTrue(torch.isfinite(output["pred_final"]).all(), condition)
                self.assertEqual(output["attn_weights"].shape[-1], len(names) + 1)


if __name__ == "__main__":
    unittest.main()
