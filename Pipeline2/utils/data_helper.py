"""Load SYMILE-MIMIC inputs and clinical labels for five-finding prediction.

Label mask: explicit 0/1 are valid; NaN/-1 are ignored.
Modality mask order: CXR, ECG, labs. Only CXR is deliberately hidden.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


TARGETS = [
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
]
MODALITIES = ("cxr", "ecg", "labs")


class SymileMIMICDataset(Dataset):
    def __init__(self, dataset_root, split="train", cxr_missing_ratio=0.30,seed=42):

        #chech the validity of the input arguments
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test.")
        if not 0 <= cxr_missing_ratio <= 1:
            raise ValueError("cxr_missing_ratio must be between 0 and 1.")


        self.split = split
        self.cxr_missing_ratio = cxr_missing_ratio
        self.CLASSES = TARGETS.copy()
        self.rng = np.random.default_rng(seed)

        root = Path(dataset_root).expanduser().resolve()
        split_dir = root / "data_npy" / split


        # Open the preprocessed arrays without loading them all into RAM.

        self.cxr = np.load(split_dir / f"cxr_{split}.npy", mmap_mode="r")
        self.ecg = np.load(split_dir / f"ecg_{split}.npy", mmap_mode="r")
        self.labs_percentiles = np.load(
            split_dir / f"labs_percentiles_{split}.npy", mmap_mode="r"
        )
        # 1 means observed and 0 means missing.
        self.labs_observed = np.load(
            split_dir / f"labs_missingness_{split}.npy", mmap_mode="r"
        )
        array_ids = np.load(
            split_dir / f"hadm_id_{split}.npy", mmap_mode="r"
        ).reshape(-1)

        #can remove this check after the run ...just for checking the validity of the input arrays
        n = len(self.cxr)
        if n == 0:
            raise ValueError("The input arrays are empty.")
        if self.cxr.shape != (n, 3, 320, 320):
            raise ValueError("Expected CXR shape (N, 3, 320, 320).")
        if self.ecg.shape != (n, 1, 5000, 12):
            raise ValueError("Expected ECG shape (N, 1, 5000, 12).")
        if self.labs_percentiles.shape != (n, 50):
            raise ValueError("Expected lab percentiles shaped (N, 50).")
        if self.labs_observed.shape != (n, 50):
            raise ValueError("Expected lab indicators shaped (N, 50).")
        if len(array_ids) != n:
            raise ValueError("Admission IDs and inputs have different lengths.")
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



        # Fix the hidden CXR samples for validation/test only.
        # Training samples receive a fresh random decision in __getitem__.

        self.fixed_cxr_available = np.ones(len(self.indices), dtype=bool)
        if split != "train":
            n_hidden = round(cxr_missing_ratio * len(self.indices))
            hidden_indices = self.rng.permutation(len(self.indices))[:n_hidden]
            self.fixed_cxr_available[hidden_indices] = False

    def __len__(self):
        return len(self.indices)

    #retrieves one adm's input ,labels and masks.
    #Dataloader calls it to collect a batch of samples.
    def __getitem__(self, index):
        row = int(self.indices[index])

        if self.split == "train":
            cxr_available = self.rng.random() >= self.cxr_missing_ratio
        else:
            cxr_available = bool(self.fixed_cxr_available[index])

        # torch.tensor copies each selected slice into writable tensor memory.
        #an availiable CXR is a 3x320x320 tensor, otherwise a zero tensor of the same shape.
        if cxr_available:
            cxr = torch.tensor(self.cxr[row], dtype=torch.float32)
        else:
            cxr = torch.zeros((3, 320, 320), dtype=torch.float32)

        ecg = torch.tensor(self.ecg[row], dtype=torch.float32)
        labs_percentiles = torch.tensor(self.labs_percentiles[row], dtype=torch.float32)
        labs_observed = torch.tensor(self.labs_observed[row], dtype=torch.float32)
        labs = torch.cat([labs_percentiles, labs_observed])

        return {
            "cxr": cxr,
            "ecg": ecg,
            "labs": labs,
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
            "label_mask": torch.tensor(self.label_masks[index], dtype=torch.bool),
            "modality_mask": torch.tensor([cxr_available, True, True], dtype=torch.bool),
            "hadm_id": torch.tensor(int(self.hadm_ids[index]), dtype=torch.long),
        }


def create_dataloaders(dataset_root, batch_size=16, train_missing_ratio=0.30,
                       val_missing_ratio=0.30, test_missing_ratio=0.30,
                       num_workers=0, seed=42):
    """Wrap the existing splits in batch loaders, without splitting again."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers != 0:
        raise ValueError("Use num_workers=0 in this introductory implementation.")

    train_dataset = SymileMIMICDataset(
        dataset_root, split="train", cxr_missing_ratio=train_missing_ratio, seed=seed
    )
    val_dataset = SymileMIMICDataset(
        dataset_root, split="val", cxr_missing_ratio=val_missing_ratio, seed=seed
    )
    test_dataset = SymileMIMICDataset(
        dataset_root, split="test", cxr_missing_ratio=test_missing_ratio, seed=seed
    )

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
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    return train_loader, val_loader, test_loader
