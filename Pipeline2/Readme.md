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

Train one fresh model with CXR, ECG and labs available for every training and
validation admission. The maximum is 50 total epochs, with early stopping after
5 validation checks without improvement in macro average precision (AP).

```bash
conda activate symile-drfuse
python Pipeline2/main.py --batch_size 8
```

Outputs go to a new version under `Pipeline2/runs_complete`. Previous runs remain
under `runs`. Start this protocol from scratch: checkpoints trained with random
CXR hiding are rejected by `--resume` and `--eval_checkpoint`.

Save both the best validation checkpoint and `last.ckpt` after completed epochs.
To resume the **new** protocol, keep `--epochs` as the total desired epoch count:

```bash
python Pipeline2/main.py --batch_size 8 --epochs 50 \
  --resume Pipeline2/runs_complete/version_0/checkpoints/last.ckpt
```

Use the same architecture, seed, optimizer and loss settings when resuming.
Work after the latest saved checkpoint may need to be repeated.

## Three test conditions

After training, the same best checkpoint is reloaded and tested three ways,
always using the final prediction head:

| Condition | Inputs returned by the test loader | Attention tokens |
|---|---|---|
| `all` | CXR, ECG, labs | ECG distinct, labs distinct, shared, CXR distinct |
| `cxr_only` | CXR | Shared, CXR distinct |
| `ecg_labs` | ECG, labs | ECG distinct, labs distinct, shared |

All conditions retain the same 464 clinical test admissions, targets and label
masks. The loader still filters the original retrieval rows using `label == 1`.
No inputs are randomly hidden in training, validation or testing. Excluded test
modalities have no input key or zero placeholder: their arrays are not opened,
their encoders are not called, and their representations do not enter fusion or
attention. A boolean availability field in model results is metadata only.

To repeat all three evaluations without training, use a checkpoint produced by
this protocol:

```bash
python Pipeline2/main.py --batch_size 8 \
  --eval_checkpoint Pipeline2/runs_complete/version_0/checkpoints/best.ckpt
```

Use the actual best filename if Lightning has added a suffix such as `best-v1`.
Each condition writes `test_results_<condition>.pt` containing final predictions,
admission IDs, targets, label masks, available features and named attention tokens.
The saved payload records the exact checkpoint path. The run also contains:

- `test_summary.csv`: macro AP/AUROC and eligible finding counts by condition.
- `test_per_finding.csv`: per-finding scores and original test label counts.
- `test_differences.csv`: paired score differences between conditions.
- `label_counts.csv`: positive, negative, uncertain and unreported counts for
  train, validation and clinical test (test only in evaluation-only runs).

Scores are AP, rather than trapezoidal PR-AUC. New logs use `Val_AP` and
`val_AP_avg_over_dxs/final`; AUROC naming and metric definitions remain unchanged.
Undefined finding scores are excluded from macro averages. The same eligible
labels are used across the three conditions.

`No Finding` has only positive and unreported entries in this release. With our
masking policy it has no negative supervision: its AP is 1 when positives exist,
and its AUROC is undefined. Macro AUROC excludes undefined classes; macro AP
includes its AP. Interpret these metrics accordingly.

## Verify

Use `dataset/symile_mimic_data.ipynb` to inspect labels and the clinical test rows,
and `dataset/dataloader_check.ipynb` to verify complete inputs and the three test
views. `dataset/symile_dataloader.ipynb` is a short example of the same loader.
`model_notebooks/main_model.ipynb` exercises all three forward paths.
`model_notebooks/training_results.ipynb` shows validation history, the test
comparison and label counts; set `RUN_DIR` to the desired run. It also recognizes
the legacy PRAUC log names in previous runs.

Use `model_notebooks/evaluate_best_checkpoint.ipynb` to load a best checkpoint
and run testing directly, with a separate result cell for all inputs, CXR only,
and ECG + labs. Each cell displays macro AP/AUROC and all 14 per-finding scores.
Set `CHECKPOINT_PATH` in its setup cell; no training or existing result CSVs are
required.

```bash
python -m unittest discover -s Pipeline2/tests -v
```

These checks use synthetic admissions, including tests with excluded input files
physically absent, encoder-call checks, checkpoint reloads and a short training
cycle. They do not train the full clinical dataset.

The implementation plan is in `docs/modality_evaluation_plan.docx`; the updated
flow is in `docs/modality_flow.mmd`. Training once and withholding inputs at test
time measures the same model's response to unavailable inputs. It does not train
specialist models or establish equivalence in naturally missing CXR populations.
