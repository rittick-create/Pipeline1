"""Load SYMILE-MIMIC inputs and all 14 CheXpert labels.

Label mask: explicit 0/1 are valid; NaN/-1 are ignored.
Train and validation use every modality. Test views omit excluded input keys.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .ecg_preprocessing import preprocess_ecg


TARGETS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
MODALITIES = ("cxr", "ecg", "labs")
TEST_CONDITIONS = {
    "all": MODALITIES,
    "cxr_only": ("cxr",),
    "ecg_labs": ("ecg", "labs"),
}


class SymileMIMICDataset(Dataset):
    def __init__(self, dataset_root, split="train", modalities=MODALITIES):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test.")
        if isinstance(modalities, str):
            raise ValueError("Pass modality names as a tuple, such as ('cxr',).")
        modalities = tuple(modalities)
        if (not modalities or len(set(modalities)) != len(modalities)
                or not set(modalities).issubset(MODALITIES)):
            raise ValueError("Choose one or more distinct modalities: cxr, ecg, labs.")
        self.modalities = tuple(name for name in MODALITIES if name in modalities)
        if split != "test" and self.modalities != MODALITIES:
            raise ValueError("Training and validation require all three modalities.")

        self.split = split
        self.CLASSES = TARGETS.copy()

        root = Path(dataset_root).expanduser().resolve()
        split_dir = root / "data_npy" / split


        # IDs are always needed; excluded modality files are never opened.
        array_ids = np.load(
            split_dir / f"hadm_id_{split}.npy", mmap_mode="r"
        ).reshape(-1)

        n = len(array_ids)
        if n == 0:
            raise ValueError("The input arrays are empty.")
        if "cxr" in self.modalities:
            self.cxr = np.load(split_dir / f"cxr_{split}.npy", mmap_mode="r")
            if self.cxr.shape != (n, 3, 320, 320):
                raise ValueError("Expected CXR shape (N, 3, 320, 320).")
        if "ecg" in self.modalities:
            self.ecg = np.load(split_dir / f"ecg_{split}.npy", mmap_mode="r")
            if self.ecg.shape != (n, 1, 5000, 12):
                raise ValueError("Expected ECG shape (N, 1, 5000, 12).")
        if "labs" in self.modalities:
            self.labs_percentiles = np.load(
                split_dir / f"labs_percentiles_{split}.npy", mmap_mode="r"
            )
            # Despite the filename, 1 means observed and 0 means missing.
            self.labs_observed = np.load(
                split_dir / f"labs_missingness_{split}.npy", mmap_mode="r"
            )
            if self.labs_percentiles.shape != (n, 50) or self.labs_observed.shape != (n, 50):
                raise ValueError("Expected lab values and indicators shaped (N, 50).")
            if not np.isin(self.labs_observed, [0, 1]).all():
                raise ValueError("Lab indicators must contain only 0 and 1.")

        # Read clinical targets and check their alignment with the inputs.
        # This is needed as we are building the dataset from the CSV files and the NPY arrays.
        # The CSV files contain the labels and the NPY arrays contain the input data.

        metadata = pd.read_csv(root / f"{split}.csv")
        required = ["subject_id", "hadm_id"] + TARGETS
        for column in required:
            if column not in metadata.columns:
                raise ValueError(f"Missing CSV column: {column}")
        if metadata[["subject_id", "hadm_id"]].isna().any().any():
            raise ValueError("Patient and admission IDs cannot be missing.")
        if not np.array_equal(metadata["hadm_id"].to_numpy(), array_ids):
            raise ValueError("CSV and NPY admission IDs differ in count or order.")

        # Keep the original array position before filtering any rows.
        metadata = metadata.copy()
        metadata["array_index"] = np.arange(len(metadata))

        if split == "test":
            # Retrieval label 1 identifies the original matching admission we need.
            required = ["label", "label_hadm_id"]
            for column in required:
                if column not in metadata.columns:
                    raise ValueError(f"Test CSV is missing: {column}")

            metadata = metadata.loc[metadata["label"] == 1].copy()

            matching_ids = metadata["hadm_id"] == metadata["label_hadm_id"]
            if not matching_ids.all():
                raise ValueError("Positive retrieval rows have mismatched IDs.")

        # Training and validation retain all rows.
        if metadata.empty:
            raise ValueError("No admissions were selected.")
        if metadata["hadm_id"].duplicated().any():
            raise ValueError("Selected admissions contain duplicates.")

        self.metadata = metadata.reset_index(drop=True)

        # Map each selected sample back to its original NPY row.
        self.indices = self.metadata["array_index"].to_numpy()
        self.hadm_ids = self.metadata["hadm_id"].to_numpy()
        self.subject_ids = self.metadata["subject_id"].to_numpy()


        # Prepare the clinical labels and their masks. The labels are 0, 1, -1, or NaN.
        # The label mask is True for 0/1 and False for -1/NaN. The labels are set to 0 where the mask is False.

        self.raw_labels = self.metadata[TARGETS].to_numpy(dtype=np.float32)
        allowed = np.isnan(self.raw_labels) | np.isin(self.raw_labels, [-1, 0, 1])
        if not allowed.all():
            raise ValueError("Clinical labels must be 0, 1, -1, or NaN.")
        self.label_masks = np.isin(self.raw_labels, [0, 1])
        self.labels = np.where(self.label_masks, self.raw_labels, 0).astype(np.float32)



    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """Return available inputs and targets for one admission, without padding."""
        row = int(self.indices[index])
        sample = {
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
            "label_mask": torch.tensor(self.label_masks[index], dtype=torch.bool),
            "hadm_id": torch.tensor(int(self.hadm_ids[index]), dtype=torch.long),
        }
        # torch.tensor copies each memory-mapped slice to writable tensor memory.
        if "cxr" in self.modalities:
            sample["cxr"] = torch.tensor(self.cxr[row], dtype=torch.float32)
        if "ecg" in self.modalities:
            # SYMILE (1, 5000, 12) -> filtered ECGFounder (12, 5000).
            sample["ecg"] = torch.from_numpy(preprocess_ecg(self.ecg[row]))
        if "labs" in self.modalities:
            values = torch.tensor(self.labs_percentiles[row], dtype=torch.float32)
            observed = torch.tensor(self.labs_observed[row], dtype=torch.float32)
            sample["labs"] = torch.cat([values, observed])
        return sample

    def label_counts(self):
        """Count original labels, before replacing unknown targets with zeros."""
        return pd.DataFrame({
            "finding": TARGETS,
            "positive": (self.raw_labels == 1).sum(axis=0),
            "negative": (self.raw_labels == 0).sum(axis=0),
            "uncertain": (self.raw_labels == -1).sum(axis=0),
            "unreported": np.isnan(self.raw_labels).sum(axis=0),
        })


def create_test_dataloaders(dataset_root, batch_size=8, num_workers=0, seed=42):
    """Three views of the same test admissions, with only selected input keys."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers != 0:
        raise ValueError("Use num_workers=0 in this introductory implementation.")

    return {
        condition: DataLoader(
            SymileMIMICDataset(dataset_root, split="test", modalities=modalities),
            batch_size=batch_size, shuffle=False, num_workers=0,
            generator=torch.Generator().manual_seed(seed + 2),
        )
        for condition, modalities in TEST_CONDITIONS.items()
    }


def create_dataloaders(dataset_root, batch_size=8, num_workers=0, seed=42):
    """Return complete train/val loaders and a dictionary of three test loaders."""
    test_loaders = create_test_dataloaders(dataset_root, batch_size, num_workers, seed)
    train_dataset = SymileMIMICDataset(dataset_root, split="train")
    val_dataset = SymileMIMICDataset(dataset_root, split="val")
    test_dataset = test_loaders["all"].dataset

    # Verify the supplied patient separation; do not move or drop patients.
    train_patients = set(train_dataset.subject_ids)
    val_patients = set(val_dataset.subject_ids)
    test_patients = set(test_dataset.subject_ids)
    if train_patients & val_patients or train_patients & test_patients or val_patients & test_patients:
        raise ValueError("Patient overlap detected between dataset splits.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    return train_loader, val_loader, test_loaders
