"""Aligned raw-modality batches for reversed PACX; no embedding caches."""
from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.transforms import functional as TF
import lightning.pytorch as pl
from .ecg_preprocessing import ECGPreprocessor


def read_metadata(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'hadm_id', 'subject_id'}.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: expected hadm_id and subject_id columns')
        rows = [(int(row['hadm_id']), int(row['subject_id'])) for row in reader]
    mapping = dict(rows)
    if not rows or len(mapping) != len(rows):
        raise ValueError(f'{path}: admissions must be nonempty and unique')
    return mapping


def validate_splits(metadata_dir):
    maps = {s: read_metadata(Path(metadata_dir) / f'{s}.csv')
            for s in ('train', 'val', 'test')}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(maps[a]) & set(maps[b]) or set(maps[a].values()) & set(maps[b].values()):
            raise ValueError(f'Patient/admission overlap between {a} and {b}')
    return maps


class RawSymileMIMICDataset(Dataset):
    def __init__(self, data_dir, split, admissions, cxr_normalization='imagenet',
                 ecg_lead_order='pacx_original'):
        root = Path(data_dir) / split
        self.arrays = {
            key: np.load(root / f'{stem}_{split}.npy', mmap_mode='r')
            for key, stem in [('cxr', 'cxr'), ('ecg', 'ecg'),
                              ('labs', 'labs_percentiles'),
                              ('labs_mask', 'labs_missingness'), ('hadm_id', 'hadm_id')]
        }
        n = len(self.arrays['cxr'])
        if n == 0 or any(len(a) != n for a in self.arrays.values()):
            raise ValueError(f'{split}: modality arrays must have equal nonzero row counts')
        if self.arrays['ecg'].shape[1:] != (1, 5000, 12):
            raise ValueError('Expected original Symile ECG shape [N,1,5000,12]')
        for key in ('labs', 'labs_mask'):
            if self.arrays[key].shape != (n, 50):
                raise ValueError(f'{key} must have shape [N,50]')
        if self.arrays['cxr'].ndim != 4 or self.arrays['cxr'].shape[1] != 3:
            raise ValueError('Expected CXR shape [N,3,H,W]')
        ids = np.asarray(self.arrays['hadm_id']).reshape(-1)
        if len(ids) != n or len(set(ids.tolist())) != n or set(ids.tolist()) != set(admissions):
            raise ValueError(f'{split}: array admission IDs do not match metadata uniquely')
        self.ids = ids.astype(np.int64)
        self.admissions = admissions
        if cxr_normalization not in ('imagenet', 'zero_one'):
            raise ValueError('cxr_normalization must be imagenet or zero_one')
        if ecg_lead_order not in ('pacx_original', 'standard'):
            raise ValueError('ecg_lead_order must be pacx_original or standard')
        self.cxr_normalization = cxr_normalization
        self.ecg_lead_order = ecg_lead_order
        self.ecg_preprocessor = ECGPreprocessor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        cxr = torch.tensor(np.array(self.arrays['cxr'][idx]), dtype=torch.float32)
        # Official tensors are already ImageNet-normalized. Never normalize twice.
        if self.cxr_normalization == 'zero_one':
            if cxr.min() < 0 or cxr.max() > 1:
                raise ValueError('zero_one CXR input must lie in [0,1]')
            cxr = TF.normalize(cxr, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        cxr = TF.resize(cxr, [224, 224], antialias=True)
        ecg = np.array(self.arrays['ecg'][idx, 0].T, dtype=np.float32)
        if self.ecg_lead_order == 'pacx_original':
            # Exact swap in the original notebook; verify source lead order before real runs.
            ecg = ecg[[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]]
        ecg = self.ecg_preprocessor.filter_bandpass(ecg, fs=500)
        ecg = torch.tensor(self.ecg_preprocessor.z_score_normalization(ecg), dtype=torch.float32)
        labs = torch.tensor(np.array(self.arrays['labs'][idx]), dtype=torch.float32)
        mask = torch.tensor(np.array(self.arrays['labs_mask'][idx]), dtype=torch.float32)
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Lab mask must be 1=observed, 0=missing')
        labs = torch.where(mask.bool(), labs, 0.0)
        if not all(torch.isfinite(x).all() for x in (cxr, ecg, labs)):
            raise ValueError(f'Nonfinite observed input for admission {self.ids[idx]}')
        if torch.any((labs < 0) | (labs > 1)):
            raise ValueError('Lab percentiles must lie in [0,1]')
        hadm = int(self.ids[idx])
        return {'cxr': cxr, 'ecg': ecg, 'labs': labs, 'labs_mask': mask,
                'hadm_id': hadm, 'subject_id': self.admissions[hadm]}


class SyntheticTriplets(Dataset):
    """Deterministic synthetic inputs, exclusively for software checks."""
    def __init__(self, size=8, seed=42):
        self.size, self.seed = size, seed

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(self.seed + idx)
        return {'cxr': torch.randn(3, 224, 224, generator=generator),
                'ecg': torch.randn(12, 5000, generator=generator),
                'labs': torch.rand(50, generator=generator),
                'labs_mask': (torch.rand(50, generator=generator) > 0.2).float(),
                'hadm_id': self.seed + idx, 'subject_id': self.seed + idx}


class Stage2DataModule(pl.LightningDataModule):
    def __init__(self, data_dir=None, metadata_dir=None, batch_size=8, num_workers=0,
                 synthetic=False, cxr_normalization='imagenet', ecg_lead_order='pacx_original'):
        super().__init__()
        if batch_size < 2:
            raise ValueError('Use batch_size >= 2 for contrastive matching')
        self.save_hyperparameters()

    def setup(self, stage=None):
        if self.hparams.synthetic:
            self.train_ds = SyntheticTriplets(max(8, self.hparams.batch_size * 2))
            self.val_ds = SyntheticTriplets(max(4, self.hparams.batch_size), seed=1000)
        else:
            maps = validate_splits(self.hparams.metadata_dir)
            def dataset(split):
                return RawSymileMIMICDataset(
                    self.hparams.data_dir, split, maps[split],
                    self.hparams.cxr_normalization, self.hparams.ecg_lead_order)
            self.train_ds, self.val_ds = dataset('train'), dataset('val')
        if len(self.train_ds) < self.hparams.batch_size:
            raise ValueError('Training set is smaller than one batch')
        if len(self.val_ds) < self.hparams.batch_size:
            raise ValueError('Validation set is smaller than one full contrastive batch')

    def _loader(self, dataset, training):
        sampler = None
        world_size = self.trainer.world_size if self.trainer is not None else 1
        global_rank = self.trainer.global_rank if self.trainer is not None else 0
        if world_size > 1:
            if not training and len(dataset) % world_size:
                raise ValueError('For DDP validation, split size must divide world_size; '
                                 'use one device to avoid duplicate contrastive negatives.')
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=global_rank,
                shuffle=training, drop_last=training)
        return DataLoader(dataset, batch_size=self.hparams.batch_size,
                          shuffle=training and sampler is None, sampler=sampler,
                          num_workers=self.hparams.num_workers, drop_last=True,
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)
