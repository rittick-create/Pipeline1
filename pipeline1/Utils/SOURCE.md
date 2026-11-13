# Source provenance

PACX-MAE reference files are in ../Pacx mae original relative to pipeline1.
Models/mae.py and Utils/utils.py are copied unchanged.
Modules/mae_lit.py adapts masking argument forwarding and repeatable validation.
Models/labs_dae.py and Modules/labs_lit.py extract the original DAE and trainer.
ProjectionHead and CLIPRegressionLoss in Modules/pacx_lit.py adapt the reference loss.
Models/net1d.py is copied unchanged from PKUDigitalHealth/ECGFounder, commit 04edac7.
Its MIT license is preserved in Models/ECGFounder_LICENSE.
Data/ecg_preprocessing.py reuses the previously verified SYMILE adapter.
New stage-specific data access, reversed distillation and linear probing follow docs/pipeline1_coding_plan.docx.
