# Pipeline2: CXR, ECG, and labs

This pipeline predicts all 14 CheXpert labels, including `No Finding` and
`Support Devices`. Their order is defined by `TARGETS` in `utils/data_helper.py`.
The loader preserves explicit 0/1 labels and masks uncertain (-1) and unreported
(NaN) labels in supervised losses and metrics.

## Restore the extra labels

The downloaded split CSVs originally retained six CheXpert columns. Restore the
other eight from `symile_mimic_data.csv` by running this from the Research folder:

```bash
python Pipeline2/dataset/restore_chexpert_labels.py
```

The script backs up the original CSVs as `.csv.bak`, validates their alignment,
and adds missing label columns in place. It preserves the existing splits,
retrieval rows, row order, and `.npy` inputs. Running it again adds nothing.
For a dataset elsewhere, pass `--data_dir /path/to/symile-mimic-aws`.

The local SYMILE-MIMIC `code/create_dataset_splits.py` also retains all 14 labels
for future dataset creation. Do not rerun splitting to update existing arrays.

## Train

```bash
conda activate symile-drfuse
python Pipeline2/main.py --batch_size 8
```

Start a fresh run for 14 outputs; five-output checkpoints have incompatible
prediction and attention layers. Logs, checkpoints, and test results go to a new
directory under `Pipeline2/runs`. The trainer gets its class count from the loader.

`No Finding` has only positive and unreported entries in this release. With our
masking policy it has no negative supervision: its AP is 1 when positives exist,
and its AUROC is undefined. Macro AUROC excludes undefined classes; macro AP
includes its AP. Interpret these metrics accordingly.

## Verify

Use `dataset/symile_mimic_data.ipynb` to inspect labels and the clinical test rows,
and `dataset/dataloader_check.ipynb` to verify the current loader and modalities.
`dataset/symile_dataloader.ipynb` is the earlier input-only experiment.
The three notebooks in `model_notebooks` use 14-output prediction heads.
