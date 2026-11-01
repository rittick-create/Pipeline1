"""Verify complete-input training and true omission of test inputs.

Run: python -m unittest discover -s Pipeline2/tests -v
"""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import MainModel, MainTrainer
from utils import MODALITIES, TARGETS, TEST_CONDITIONS, SymileMIMICDataset, create_dataloaders
from utils.evaluation import save_comparison
import main as entry

torch.set_num_threads(2)


def make_fixture(root):
    """Small cohorts with real input shapes, retrieval repeats and unknown labels."""
    for split, ids in [("train", [10, 11]), ("val", [20, 21]),
                       ("test", [30, 30, 31, 31])]:
        n = len(ids)
        folder = root / "data_npy" / split
        folder.mkdir(parents=True)
        np.save(folder / f"hadm_id_{split}.npy", ids)
        for name, shape in [("cxr", (n, 3, 320, 320)), ("ecg", (n, 1, 5000, 12)),
                            ("labs_percentiles", (n, 50)), ("labs_missingness", (n, 50))]:
            a = np.ones(shape, dtype=np.float32)
            if name != "labs_missingness":
                a *= np.arange(1, n + 1).reshape(n, *([1] * (len(shape) - 1))) / n
            np.save(folder / f"{name}_{split}.npy", a)
        frame = pd.DataFrame({"subject_id": np.array(ids) * 10, "hadm_id": ids})
        for i, name in enumerate(TARGETS):
            frame[name] = np.arange(n) % 2
        frame[TARGETS[-1]] = np.nan
        frame[TARGETS[-2]] = -1
        if split == "test":
            frame["label"] = [0, 1, 0, 1]
            frame["label_hadm_id"] = [31, 30, 30, 31]
        frame.to_csv(root / f"{split}.csv", index=False)


class SmallInputs(Dataset):
    """Crop synthetic inputs only, to exercise trainer orchestration quickly."""
    def __init__(self, original):
        self.original = original

    def __len__(self):
        return len(self.original)

    def __getitem__(self, i):
        item = self.original[i]
        if "ecg" in item:
            item["ecg"] = item["ecg"][:, :16]
        if "cxr" in item:
            item["cxr"] = item["cxr"][:, :64, :64]
        return item

    def label_counts(self):
        return self.original.label_counts()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_views_share_admissions_labels_and_original_array_positions(self):
        train, val, tests = create_dataloaders(self.root, batch_size=2)
        for loader in [train, val]:
            self.assertEqual(set(MODALITIES) & set(loader.dataset[0]), set(MODALITIES))
            self.assertNotIn("modality_mask", loader.dataset[0])
        reference = tests["all"].dataset
        np.testing.assert_array_equal(reference.indices, [1, 3])
        for condition, modalities in TEST_CONDITIONS.items():
            ds = tests[condition].dataset
            self.assertEqual(set(ds[0]) & set(MODALITIES), set(modalities))
            np.testing.assert_array_equal(ds.hadm_ids, reference.hadm_ids)
            np.testing.assert_array_equal(ds.labels, reference.labels)
            np.testing.assert_array_equal(ds.label_masks, reference.label_masks)
            self.assertFalse(ds.label_masks[:, -2:].any())
            self.assertTrue((ds.labels[:, -2:] == 0).all())
            counts = ds.label_counts()
            self.assertTrue(counts[["positive", "negative", "uncertain", "unreported"]].sum(axis=1).eq(2).all())
            if "cxr" in modalities:
                self.assertTrue(torch.equal(ds[0]["cxr"], torch.full((3, 320, 320), .5)))

    def test_excluded_modality_files_do_not_need_to_exist(self):
        folder = self.root / "data_npy/test"
        (folder / "cxr_test.npy").unlink()
        ds = SymileMIMICDataset(self.root, "test", ("ecg", "labs"))
        self.assertNotIn("cxr", ds[0])
        self.assertFalse(hasattr(ds, "cxr"))
        # Create a CXR file again, then physically omit ECG and both lab files.
        np.save(folder / "cxr_test.npy", np.ones((4, 3, 320, 320), np.float32))
        for name in ["ecg", "labs_percentiles", "labs_missingness"]:
            (folder / f"{name}_test.npy").unlink()
        ds = SymileMIMICDataset(self.root, "test", ("cxr",))
        self.assertEqual(set(ds[0]) & set(MODALITIES), {"cxr"})
        self.assertFalse(hasattr(ds, "ecg"))
        self.assertFalse(hasattr(ds, "labs_percentiles"))

    def test_partial_training_and_empty_input_selection_are_rejected(self):
        for split in ["train", "val"]:
            with self.assertRaises(ValueError):
                SymileMIMICDataset(self.root, split, ("cxr",))
        for inputs in [(), ("cxr", "cxr"), ("bad",), "cxr"]:
            with self.assertRaises(ValueError):
                SymileMIMICDataset(self.root, "test", inputs)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model = MainModel(hidden_size=8, ecg_n_head=2, ecg_dropout=0, labs_dropout=0).eval()
        cls.inputs = {"cxr": torch.randn(2, 3, 64, 64),
                      "ecg": torch.randn(2, 1, 16, 12), "labs": torch.randn(2, 100)}

    def test_dynamic_attention_and_no_calls_to_excluded_encoders(self):
        def digest():
            h = hashlib.sha256()
            for value in self.model.state_dict().values():
                h.update(value.numpy().tobytes())
            return h.digest()
        before = digest()
        with torch.inference_mode():
            with patch.object(self.model.cxr_model_feat, "forward", side_effect=AssertionError("CXR used")):
                out = self.model(ecg=self.inputs["ecg"], labs=self.inputs["labs"])
                self.assertFalse(any("cxr" in key for key in out))
                self.assertEqual(out["attn_token_names"], ("ecg_distinct", "labs_distinct", "shared"))
            with patch.object(self.model.ecg_model, "forward", side_effect=AssertionError("ECG used")), \
                 patch.object(self.model.labs_model, "forward", side_effect=AssertionError("Labs used")):
                out = self.model(cxr=self.inputs["cxr"])
                self.assertEqual(out["attn_token_names"], ("shared", "cxr_distinct"))
                torch.testing.assert_close(out["feat_avg_shared"], out["feat_cxr_shared"])
            for condition, names in TEST_CONDITIONS.items():
                out = self.model(**{name: self.inputs[name] for name in names})
                self.assertEqual(out["pred_final"].shape, (2, 14))
                self.assertEqual(out["attn_weights"].shape, (2, 14, len(names) + 1))
                self.assertTrue(torch.isfinite(out["pred_final"]).all())
                torch.testing.assert_close(out["attn_weights"].sum(-1), torch.ones(2, 14))
        self.assertEqual(before, digest())  # Includes BatchNorm buffers.
        with self.assertRaises(ValueError):
            self.model()

    def test_complete_input_fusion_matches_reference_definition(self):
        with torch.inference_mode():
            out = self.model(**self.inputs)
            h = torch.stack([out[f"feat_{m}_shared"] for m in MODALITIES], 1)
            mean_probability = h.sigmoid().mean(1)
            expected = torch.logit(mean_probability)
            torch.testing.assert_close(out["feat_avg_shared"], expected, atol=2e-6, rtol=2e-6)


class TrainerTests(unittest.TestCase):
    def test_complete_train_checkpoint_three_tests_and_repeat_evaluation(self):
        # Real encoders and losses, with small synthetic spatial/time dimensions.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            train, val, tests = create_dataloaders(root, batch_size=2)
            cropped = lambda dl: DataLoader(SmallInputs(dl.dataset), batch_size=2)
            loaders = (cropped(train), cropped(val), {k: cropped(v) for k, v in tests.items()})
            args = entry.parse_args(["--data_dir", str(root), "--output_dir", str(root / "runs"),
                                     "--hidden_size", "8", "--ecg_n_head", "2",
                                     "--dropout", "0", "--epochs", "1", "--batch_size", "2"])
            with patch.object(entry, "create_dataloaders", return_value=loaders):
                entry.main(args)
            run = root / "runs/version_0"
            self.assertTrue((run / "checkpoints/last.ckpt").exists())
            best = run / "checkpoints/best.ckpt"
            entry.validate_checkpoint(best, args)
            saved = {k: torch.load(run / f"test_results_{k}.pt", weights_only=False)["test_results"]
                     for k in TEST_CONDITIONS}
            for name, result in saved.items():
                self.assertEqual(tuple(result["modalities"]), TEST_CONDITIONS[name])
                self.assertEqual(result["preds"].shape, (2, 14))
                self.assertEqual(result["attn_weight"].shape[-1], len(TEST_CONDITIONS[name]) + 1)
                self.assertFalse(any(f"feat_{m}_shared" in result for m in set(MODALITIES) - set(result["modalities"])))
            summary = pd.read_csv(run / "test_summary.csv")
            self.assertEqual(summary.condition.tolist(), list(TEST_CONDITIONS))
            counts = pd.read_csv(run / "label_counts.csv")
            self.assertEqual(len(counts), 42)
            self.assertEqual(len(pd.read_csv(run / "test_per_finding.csv")), 14)
            # Saving must reject comparisons that silently change the cohort.
            bad = {k: dict(v) for k, v in saved.items()}
            bad["cxr_only"]["hadm_ids"] = torch.tensor([99, 100])
            with self.assertRaises(ValueError):
                save_comparison(bad, tests["all"].dataset.label_counts(), run)
            # Evaluation-only uses the same checkpoint and does not create train loaders.
            args.eval_checkpoint = best
            with patch.object(entry, "create_test_dataloaders", return_value=loaders[2]), \
                 patch.object(entry, "create_dataloaders", side_effect=AssertionError("Training called")):
                entry.main(args)
            repeat = root / "runs/version_1"
            for name in TEST_CONDITIONS:
                again = torch.load(repeat / f"test_results_{name}.pt", weights_only=False)["test_results"]
                torch.testing.assert_close(again["preds"], saved[name]["preds"])
                torch.testing.assert_close(again["hadm_ids"], saved[name]["hadm_ids"])
            # max_epochs is a total; resuming one completed epoch adds only one.
            args.eval_checkpoint = None
            args.resume = run / "checkpoints/last.ckpt"
            args.epochs = 2
            with patch.object(entry, "create_dataloaders", return_value=loaders):
                entry.main(args)
            resumed = torch.load(root / "runs/version_2/checkpoints/last.ckpt",
                                 weights_only=False, mmap=True)
            self.assertEqual(resumed["epoch"], 1)
            self.assertEqual(resumed["global_step"], 2)

    def test_unknown_labels_and_undefined_metrics(self):
        # Invoke the metric helper with a lightweight holder; no encoders needed.
        from types import SimpleNamespace
        holder = SimpleNamespace(label_names=["both_classes", "positive_only", "unknown"])
        scores = torch.tensor([[.1, .2, .9], [.9, .4, .1], [.99, .6, .5]])
        labels = torch.tensor([[0., 1., 0.], [1., 1., 0.], [0., 1., 0.]])
        masks = torch.tensor([[True, True, False], [True, True, False], [False, True, False]])
        ap, roc = MainTrainer._compute_masked_metrics(holder, scores, labels, masks)
        torch.testing.assert_close(ap[:2], torch.ones(2))
        self.assertEqual(roc[0].item(), 1)
        self.assertTrue(torch.isnan(roc[1:]).all())
        self.assertTrue(torch.isnan(ap[2]))


if __name__ == "__main__":
    unittest.main()
