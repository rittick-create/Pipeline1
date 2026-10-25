"""Restore all CheXpert columns without recreating splits or input arrays."""

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_helper import TARGETS


def restore_labels(dataset_root):
    root = Path(dataset_root).expanduser().resolve()
    source = pd.read_csv(
        root / "symile_mimic_data.csv",
        usecols=["subject_id", "hadm_id", "cxr_path", *TARGETS],
        float_precision="round_trip",
    ).set_index("hadm_id", verify_integrity=True)

    # Validate every split before writing any files.
    updates = []
    for split in ("train", "val", "val_retrieval", "test"):
        path = root / f"{split}.csv"
        metadata = pd.read_csv(path, float_precision="round_trip")
        if not metadata["hadm_id"].isin(source.index).all():
            raise ValueError(f"{split}: admissions are missing from the source CSV.")
        array_ids = np.load(
            root / "data_npy" / split / f"hadm_id_{split}.npy", mmap_mode="r"
        ).reshape(-1)
        if not np.array_equal(metadata["hadm_id"].to_numpy(), array_ids):
            raise ValueError(f"{split}: CSV and NPY admission order differs.")

        # Reindex rather than merge: repeated retrieval rows retain their order.
        aligned = source.reindex(metadata["hadm_id"]).reset_index(drop=True)
        existing = ["subject_id", "cxr_path"] + [c for c in TARGETS if c in metadata]
        pd.testing.assert_frame_equal(
            metadata[existing], aligned[existing], check_dtype=False, check_exact=True
        )
        missing = [c for c in TARGETS if c not in metadata]
        restored = metadata.copy()
        for column in missing:
            restored[column] = aligned[column]
        updates.append((path, restored, missing))

    for path, metadata, missing in updates:
        if missing:
            backup = path.with_suffix(".csv.bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            temporary = path.with_suffix(".csv.tmp")
            metadata.to_csv(temporary, index=False)
            temporary.replace(path)
        print(f"{path.name}: {len(metadata)} rows, 14 label columns; added {len(missing)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", type=Path,
        default=Path(__file__).resolve().parents[2] / "symile-mimic" / "symile-mimic-aws",
    )
    restore_labels(parser.parse_args().data_dir)
