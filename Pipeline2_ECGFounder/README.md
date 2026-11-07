# Pipeline2 with ECGFounder

This is a separate version of Pipeline2. The original folder and its checkpoints
are unchanged. The CXR encoder, labs MLP, fusion, losses, label masks and three
test conditions are retained. The ECG Transformer is replaced by the pretrained
12-lead ECGFounder Net1D backbone and two new independent shared/distinct branches.

## Data and weights

The default data directory is `../symile-mimic/symile-mimic-aws`. No dataset is
copied or modified. The default pretrained weights are
`../ecgfounder/checkpoint/12_lead_ECGFounder.pth`; override with `--ecg_checkpoint`.
The upstream Net1D implementation and MIT license are in `third_party/ecgfounder`.

The loader converts each stored ECG from `(1, 5000, 12)` to `(12, 5000)`, reorders
aVF/aVL, applies a 50 Hz notch, 0.67–40 Hz bandpass, median baseline removal and
global record z-score. It does not repeat SYMILE's min-max scaling or resample.
This adaptation matched the raw-record reference on the 21 checked records;
that check does not establish equivalence for every record or model accuracy.
Preprocessing runs on demand on the CPU; the source NPY arrays are unchanged.

ECGFounder produces 1,024 features. Each new branch maps these to 256 features
by default; the distinct branch also predicts the 14 findings for its auxiliary
loss. The original ECGFounder's 150-output classifier is discarded.

## Run

Use the existing `symile-drfuse` conda environment. From this folder:

```bash
conda activate symile-drfuse
python main.py
```

By default the pretrained ECG backbone is fine-tuned at `1e-5`; all other
trainable layers use `1e-4`. To train new branches while keeping the pretrained
backbone frozen, use `python main.py --freeze_ecg`. This is a separate experiment;
do not change the freeze setting when resuming an existing run. Batch size is 8,
maximum epochs 50 and early-stopping patience 5. A smaller batch can be selected
with `--batch_size` if needed; the old model's batch-size memory limit does not
establish the limit for this architecture.

Training and validation use all modalities. The best validation macro-AP
checkpoint is tested on all modalities, CXR only, and ECG + labs, using the same
test admissions and final prediction head. Excluded inputs are never loaded or
encoded. Unknown labels (NaN/−1) remain excluded from losses and metrics.

Results go to this version's `runs_complete/version_N`, including checkpoints,
label counts, prediction tensors, summary, differences and per-finding AP/AUROC.

Resume this version with its own last checkpoint (epochs is the total limit):

```bash
python main.py --resume runs_complete/version_0/checkpoints/last.ckpt
```

Evaluate this version's best checkpoint without training:

```bash
python main.py --eval_checkpoint runs_complete/version_0/checkpoints/best.ckpt
```

The official pretrained file initializes ECGFounder. Our own best checkpoint is
selected after training. Old Pipeline2 checkpoints contain a different ECG
architecture and cannot be resumed here. Training is not started by setup.
