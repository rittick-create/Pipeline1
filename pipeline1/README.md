# Pipeline 1 — checkpoints

- [ ] **1. Set up Stage 1:** Reuse PACX-MAE's CXR encoder, lab encoder, and ECGFounder in your pipeline.
- [ ] **2. Prepare model weights:** Load pretrained weights when available; otherwise plan to train the CXR and lab encoders when their data is available.
- [ ] **3. Check the basic flow:** Use dummy inputs to confirm all three encoders produce embeddings. This checks the setup, not research performance.
- [ ] **4. Build Stage 2:** Combine ECG and lab embeddings to predict the frozen CXR encoder's embedding. Add contrastive and prediction losses.
- [ ] **5. Connect the dataset:** When Symile-MIMIC is available, replace dummy inputs with correctly paired ECG, labs, and CXR samples using its train/validation/test splits.
- [ ] **6. Train the pipeline:** Complete any missing Stage 1 training, then train Stage 2 with the CXR encoder frozen.
- [ ] **7. Evaluate downstream tasks:** Use ECG + labs and compare performance with and without Stage 2 training.

**Start now:** checkpoints 1–4. Dummy inputs let you build the flow while waiting for data; meaningful training and evaluation come later.
