# Pipeline 1 — checkpoints

- [ ] **1. Set up Stage 1:** Reuse PACX-MAE's CXR encoder, lab encoder, and ECGFounder in your pipeline.
- [ ] **2. Prepare model weights:** Load pretrained weights when available; otherwise plan to train the CXR and lab encoders when their data is available.
- [ ] **3. Check the basic flow:** Use dummy inputs to confirm all three encoders produce embeddings. This checks the setup, not research performance.
- [ ] **4. Build Stage 2:** Combine ECG and lab embeddings to predict the frozen CXR encoder's embedding. Add contrastive and prediction losses.
- [ ] **5. Connect the dataset:** When Symile-MIMIC is available, replace dummy inputs with correctly paired ECG, labs, and CXR samples using its train/validation/test splits.
- [ ] **6. Train the pipeline:** Complete any missing Stage 1 training, then train Stage 2 with the CXR encoder frozen.
- [ ] **7. Evaluate downstream tasks:** Use ECG + labs and compare performance with and without Stage 2 training.

**Start now:** checkpoints 1–4. Dummy inputs let you build the flow while waiting for data; meaningful training and evaluation come later.

## Reversed Stage 2 implementation

`Stage2_REVERSE.py` is the separate training entry point. The original copied `Stage2_PACX.py` and `Modules/pacx_lit.py` remain unchanged as references. The reversed model and raw-input loader are in `Modules/reverse_pacx_lit.py` and `Data/stage2_data.py`. The ECGFounder classes and lab DAE definition are extracted into importable files without the notebooks' data-loading/training side effects.

For software checks only, run from `pipeline1/`:

```bash
python Stage2_REVERSE.py --synthetic --allow-random-init --max-epochs 1 --batch-size 4 --num-workers 0 --accelerator gpu --devices 1 --save-dir /tmp/pipeline1-stage2-smoke
```

This uses random frozen CXR targets and cannot establish research performance. Real training requires all three compatible checkpoints and Symile-MIMIC `.npy` arrays. Supply a metadata directory with `train.csv`, `val.csv`, and `test.csv`, each containing `hadm_id,subject_id`, aligned to the official `hadm_id_<split>.npy` values. The loader verifies unique admissions and patient-disjoint splits. CXR tensors are assumed already ImageNet-normalized by default. ECG preprocessing and the original aVF/aVL channel reorder follow the PACX-MAE extraction notebook; verify the source lead convention before real training.
