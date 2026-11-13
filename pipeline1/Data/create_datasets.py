"""Stage-specific inputs. SYMILE test access is restricted to Stage 3 callers."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from .ecg_preprocessing import preprocess_ecg
from .targets import TARGETS


class ChestXrayDataset(Dataset):
    def __init__(self, root, split, image_size=224):
        root = Path(root)
        batch1 = root / "CheXpert-v1.0 batch 1 (validate & csv)"
        metadata = pd.read_csv(batch1 / f"{split}.csv")
        patients = {}
        for folder in root.glob("CheXpert-v1.0 batch *"):
            for patient in folder.glob("patient*"):
                if patient.is_dir():
                    patients[patient.name] = patient
        self.paths = []
        self.subject_ids = []
        for value in metadata["Path"]:
            patient, study, image = Path(value).parts[-3:]
            base = (
                patients.get(patient)
                if split == "train"
                else batch1 / "valid" / patient
            )
            if base is None or not (base / study / image).is_file():
                raise FileNotFoundError(f"Cannot resolve CheXpert image {value}")
            self.paths.append(base / study / image)
            self.subject_ids.append(patient)
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if split == "train":
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = T.Compose(
                [T.Resize((image_size, image_size)), T.ToTensor(), normalize]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class SymileMIMICDataset(Dataset):
    def __init__(self, root, split, stage):
        if stage not in ("labs", "distillation", "classification"):
            raise ValueError("Unknown stage")
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        if split == "test" and stage != "classification":
            raise ValueError("SYMILE test data is reserved for Stage 3.")
        self.stage = stage
        self.split = split
        root = Path(root)
        folder = root / "data_npy" / split
        # Earlier stages read no test metadata or test arrays.
        ids = np.load(folder / f"hadm_id_{split}.npy", mmap_mode="r").reshape(-1)
        self.metadata = pd.read_csv(root / f"{split}.csv")
        if not np.array_equal(ids, self.metadata.hadm_id.to_numpy()):
            raise ValueError("CSV and NPY admission order differs")
        self.metadata["array_index"] = np.arange(len(ids))
        if split == "test":
            self.metadata = self.metadata.loc[self.metadata.label.eq(1)].copy()
            if not self.metadata.hadm_id.eq(self.metadata.label_hadm_id).all():
                raise ValueError("Retrieval positives have mismatched IDs")
        self.metadata = self.metadata.reset_index(drop=True)
        if self.metadata.empty or self.metadata.hadm_id.duplicated().any():
            raise ValueError("Admissions must be nonempty and unique")
        self.indices = self.metadata.array_index.to_numpy()
        self.subject_ids = self.metadata.subject_id.to_numpy()
        self.labs = np.load(folder / f"labs_percentiles_{split}.npy", mmap_mode="r")
        self.observed = np.load(folder / f"labs_missingness_{split}.npy", mmap_mode="r")
        if self.labs.shape != (len(ids), 50) or self.observed.shape != self.labs.shape:
            raise ValueError("Expected 50 lab values and observation indicators")
        if not np.isin(self.observed, [0, 1]).all():
            raise ValueError("Expected observed=1, missing=0")
        if stage != "labs":
            self.ecg = np.load(folder / f"ecg_{split}.npy", mmap_mode="r")
            if self.ecg.shape != (len(ids), 1, 5000, 12):
                raise ValueError("Unexpected ECG array shape")
        if stage == "distillation":
            self.cxr = np.load(folder / f"cxr_{split}.npy", mmap_mode="r")
            if self.cxr.shape != (len(ids), 3, 320, 320):
                raise ValueError("Unexpected CXR array shape")
        if stage == "classification":
            self.raw_labels = self.metadata[TARGETS].to_numpy(dtype=np.float32)
            if not (
                np.isnan(self.raw_labels) | np.isin(self.raw_labels, [-1, 0, 1])
            ).all():
                raise ValueError("Unexpected target values")
            self.label_mask = np.isin(self.raw_labels, [0, 1])
            self.labels = np.where(self.label_mask, self.raw_labels, 0).astype(
                np.float32
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        x = torch.tensor(self.labs[row], dtype=torch.float32)
        mask = torch.tensor(self.observed[row], dtype=torch.float32)
        x = torch.where(mask.bool(), x, 0.0)
        if self.stage == "labs":
            return x, mask
        result = {
            "ecg": torch.from_numpy(preprocess_ecg(self.ecg[row])),
            "labs": x,
            "labs_mask": mask,
            "hadm_id": torch.tensor(int(self.metadata.hadm_id.iloc[index])),
        }
        if self.stage == "distillation":
            image = torch.tensor(self.cxr[row], dtype=torch.float32)
            result["cxr"] = TF.resize(image, [224, 224], antialias=True)
        else:
            result["labels"] = torch.from_numpy(self.labels[index].copy())
            result["label_mask"] = torch.from_numpy(self.label_mask[index].copy())
        return result

    def label_counts(self, limit=None):
        raw_labels = self.raw_labels[:limit]
        return pd.DataFrame(
            {
                "finding": TARGETS,
                "positive": (raw_labels == 1).sum(0),
                "negative": (raw_labels == 0).sum(0),
                "uncertain": (raw_labels == -1).sum(0),
                "unreported": np.isnan(raw_labels).sum(0),
            }
        )
