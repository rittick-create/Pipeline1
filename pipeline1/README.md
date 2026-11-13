# Reversed PACX MAE pipeline

Stage 1 retains PACX-MAE's ViT-B/16 MAE and lab DAE. Stage 2 freezes the CXR teacher,
trains ECGFounder's last residual block and the full lab encoder, and independently
predicts CXR embeddings from ECG and labs. Stage 3 freezes both students, concatenates
1024 + 256 features, and trains only a linear 14-finding classifier.

Run from this folder using the existing `symile-drfuse` environment.

```bash
python Stage1_MAE.py
python Stage1_Labs.py
python Stage2_PACX.py --cxr_checkpoint runs/mae/version_0/checkpoints/best.ckpt --labs_checkpoint runs/labs/version_0/checkpoints/best.ckpt
python Stage3_CL.py --stage2_checkpoint runs/distillation/version_0/checkpoints/best.ckpt
```

Use the actual best filenames printed after each run. Defaults are 400 MAE epochs,
50 lab epochs, 50 distillation epochs and 40 classifier epochs. Local batch size is 4,
workers 0, seed 42 and precision 32. The full CheXpert MAE run is substantial.
Set `--batch_size`, `--num_workers`, `--precision`, and `--epochs` for your machine.
The 90% MAE masking argument is explicitly forwarded, fixing the reference mismatch.

Each stage saves best and last checkpoints with CSV metrics and hyperparameters.
For an interrupted run, use that stage's `--resume /path/to/last.ckpt`. Epochs is the
total target. Keep saved learning rate, weight decay, seed, sample limit and loss settings.
Stage 3 supports `--eval_checkpoint /path/to/classifier/best.ckpt` for testing alone.

Stage 1 labs and Stage 2 load only SYMILE train/val. Test arrays and test metadata
are first opened by Stage 3 testing. Stage 3 train/val/test loaders use ECG and labs.
Known disease labels 0/1 are valid; -1/NaN are masked in both loss and metrics.

Use `--sample_limit 4 --epochs 1` for a software check on a few real admissions.
It is recorded in checkpoints/results. Sample-limited results are implementation
checks. Omit sample_limit for research training and final evaluation.

The current focus is the student pipeline. PACX-MAE CXR-reference reproduction and
comparison are deferred until the student's results are available.
See docs/pipeline1_coding_plan.docx and Utils/SOURCE.md for design and source provenance.

## Repository structure and ECGFounder construction

The project retains the PACX-MAE `Data`, `Models`, `Modules`, `Scripts`, `Utils`,
`Env`, `Figs` and stage-entry-point structure. `Stage1_MAE.py` follows the reference's
`main(args)` layout. The original backbone and reconstruction modules supply Stage 1.
Local adaptations handle split image batches, CSV logging, resume, and correct 90% masking.

`Models/net1d.py` contains the complete ECGFounder implementation, including padded
convolutions, residual blocks, stages and global pooling. `Models/ecg_founder.py`
constructs its feature encoder. This separates network construction from initialization.

```python
from Models.ecg_founder import ECGFounderEncoder, DEFAULT_ECG

encoder = ECGFounderEncoder()          # Build the full network with random weights.
encoder.initialize(DEFAULT_ECG)        # Load the downloaded pretrained backbone.
# features = encoder(ecg)              # [B, 12, 5000] -> [B, 1024]
```

Random construction supports architecture inspection and software experiments.
Reproducing ECGFounder's large-scale pretraining requires its original training data
and protocol. Our research pipeline starts with its downloaded pretrained weights.

In the PACX-MAE reference, inspect `Data/ecg_features_extraction.ipynb` and
`Modules/pacx_lit.py`. The original direction adapts CXR to frozen ECG/lab features.
Here, the frozen CXR embedding supervises separate ECG-to-CXR and labs-to-CXR heads.
ECGFounder's last residual block and the lab encoder adapt during Stage 2. Stage 3
freezes their learned encoders and trains only a new `Linear(1280, 14)` head.

Run implementation tests with `python -m unittest discover -s tests -v`.
Review the saved classifier run with `results_notebooks/training_results.ipynb`.
