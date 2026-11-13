"""Lightning data modules with lazy Stage 3 test loading."""

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset
from .create_datasets import ChestXrayDataset, SymileMIMICDataset


class PipelineDataModule(pl.LightningDataModule):
    def __init__(
        self, stage, root, batch_size=4, num_workers=0, seed=42, sample_limit=None
    ):
        super().__init__()
        self.stage, self.root = stage, root
        self.batch_size, self.num_workers = batch_size, num_workers
        self.seed, self.sample_limit = seed, sample_limit
        self.datasets = {}

    def _make(self, split):
        if self.stage == "mae":
            return ChestXrayDataset(self.root, "valid" if split == "val" else split)
        return SymileMIMICDataset(self.root, split, self.stage)

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            for split in ("train", "val"):
                if split not in self.datasets:
                    self.datasets[split] = self._make(split)
            if set(self.datasets["train"].subject_ids) & set(
                self.datasets["val"].subject_ids
            ):
                raise ValueError("Training and validation patients overlap")
        if stage == "test":
            if self.stage != "classification":
                raise ValueError("Only Stage 3 can use SYMILE test data")
            self.datasets["test"] = self._make("test")
            test_patients = set(self.datasets["test"].subject_ids)
            for split in ("train", "val"):
                # Check patient separation even during evaluation-only runs.
                import pandas as pd
                from pathlib import Path

                metadata = pd.read_csv(
                    Path(self.root) / f"{split}.csv", usecols=["subject_id"]
                )
                if test_patients & set(metadata.subject_id):
                    raise ValueError("Test patients overlap with fitting cohorts")

    def _loader(self, split):
        dataset = self.datasets[split]
        if self.sample_limit is not None:
            dataset = Subset(dataset, range(min(self.sample_limit, len(dataset))))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=torch.Generator().manual_seed(self.seed),
            drop_last=self.stage == "distillation" and split == "train",
        )

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")


class CXRDataModule(PipelineDataModule):
    """Named Stage 1 interface matching the reference repository."""

    def __init__(self, root, batch_size=4, num_workers=0, seed=42, sample_limit=None):
        super().__init__("mae", root, batch_size, num_workers, seed, sample_limit)
