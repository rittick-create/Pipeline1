# Original PACX-MAE code copied into pipeline1

Source: `../Pacx mae original`, commit `92ddf9d90ad99f00200383d3e3334b4775806b9e`.

Copied 60 files without editing their contents. Every copy was checked byte for byte.

- Stage 1: CXR MAE, lab DAE, ECG/lab extraction notebooks, and their supporting files.
- Data: original loading and preprocessing code, including the Symile scripts.
- Stage 2: original training script and distillation module, copied as a reference for later changes. It still trains CXR to predict ECG/lab targets.
- Stage 3 entry points and serving code have not been copied; ECG + labs inference will be implemented later.

## What remains for later

Our intended direction is ECG + labs predicting a frozen CXR embedding. The Stage 2 connections, fusion/prediction heads, encoder freezing, relevant data flow, and downstream inference still need adaptation.

Original paths, imports, dependencies, and behavior are preserved. This includes the MAE mask-ratio mismatch and the lab script loading data and starting training at import time. No fixes, notebook extraction, package installation, checkpoint downloads, or training were performed.

## Subsequent reversed Stage 2 work

The original copied Stage 2 files remain unchanged as references. Reversed Stage 2 lives in `Stage2_REVERSE.py`, `Modules/reverse_pacx_lit.py`, and `Data/stage2_data.py`; encoder and preprocessing definitions were extracted into separate import-safe files. Only synthetic-data implementation checks have run. Real checkpoint compatibility and paired-data loading still need to be confirmed once those files are available.

## Preserved drafts

- `resources/drafts/before_original_copy_20261015_185351_523232/Stage1_MAE.py`

## Copied files (SHA-256)

- `Data/__init__.py` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `Data/chestx6_data.ipynb` — `393c2dec964d80371e6319ca7765e612ed647d93390a6302bbb6795856edf3a0`
- `Data/chexchonet_data.ipynb` — `70b9d4cc9047592984e519a49997c353a55205ca322c11bf3145c2a866356497`
- `Data/chexpert_data.ipynb` — `2cc2fbace208c93e951bb3e2280844e99a84a4e06752b538675c51f10b0cb532`
- `Data/covid_data.ipynb` — `c917aa7cace29a1cc8841bbae4ea3f7696a3bddbb118e3a855b6523e34a2dd5e`
- `Data/covidqu_data.ipynb` — `d2d13bc3d579e4fd1ded198e1f5ea70e289fb9b73f6ad99b350f295d4486230b`
- `Data/covidqu_seg_data.ipynb` — `b7b72dd5eaeefe42134a792e0dcbcc6ee9e27a2b2842d6aea9b3db0546b11a66`
- `Data/create_datasets.py` — `e2468626e3c7db23c9880bb624ea48f928774c5c89f5b1eb8a13c0ab03d2bdb9`
- `Data/create_test_uni.ipynb` — `5198f8b9bb0544fe739e5be5bded3527f4bc09f373f12310e7db24c2f52fe1cb`
- `Data/data_modules.py` — `0e729de4100581412003e9009c73125691fd02e3248d7a972fb19c9555e03dff`
- `Data/ecg_features_extraction.ipynb` — `82d747ed80520e24929c289ca6b79f28970af2746c263adab06daf94fb9ba9c2`
- `Data/labs_features_extraction.ipynb` — `182d61a260ee0bf5d721860904e75c4a4371a7c02bbb38d1f3a8369e651eff03`
- `Data/medmod_data.ipynb` — `c0798bc29a13b08f7f99ed0e1a306a089e7e4112e83ba099ba7ff9c7117c001e`
- `Data/nih_data.ipynb` — `437c9cd4830de4786c55894def8e6899acd83acd30863b1f86fb975dc227485f`
- `Data/pne_data.ipynb` — `38c98bc126014f35ded8f6e6df09c483a89e0a4855f5eefc29e820c849fe7a31`
- `Data/qata_data.ipynb` — `0932eca602bbf2a467fec7b596ef15b6730000a1df84451bbdf121c986a38bbb`
- `Data/symile_mimic_data.ipynb` — `bc8de6f109456806af72aa03e3a1c2244340e70695ca5fd708dfb8826cbfda27`
- `Data/tb_data.ipynb` — `176d068049db4dfeb12795b48eb02f585ee31db5dd4115f73315e7f152d693d0`
- `Data/vindr_data.ipynb` — `26f4db4bad6370b8ee479605cee261f540f3b4d5e50e9855986de803c7e95061`
- `Env/environment.yml` — `22a0a68fe9125454c181eb7438b2d2ba690298dd8f27ea75bdd8c14027c062db`
- `Env/requirements.txt` — `257fa9edea31594f4c91316b9bdf9a2ffbec6f585d94bb1732ad8b74712f45ec`
- `Models/__init__.py` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `Models/labs_dae.py` — `27ecc3f977fc22e9c4ad5dcc0e21216f0f0e3511b06fbec3ca5b2a081a663db6`
- `Models/mae.py` — `54a19f440cce50039047744feaf7b91d18b570b1bf0b90d4fd23b50b8b180f34`
- `Models/models.py` — `ffabf380df972cde2ab2b0a7c1828c9f1aa862fe6921d4db46f75d39ac24af87`
- `Modules/__init__.py` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `Modules/mae_lit.py` — `d88643b0b2497daf81a297a90fda80534ac6f3d8da907f5f16b9d481675551b3`
- `Modules/pacx_lit.py` — `5a786a8a21b3e009c2daad9b79b2c9f28213a7b7a437f054ad9dbf3476d0a852`
- `Scripts/__init__.py` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `Scripts/chexchonet/echo_pp.py` — `cf40af390edf4bd4468d5c71f4bd95308a1c5a0ee3040d19a3ffb86a4ec1d26a`
- `Scripts/from_symile/args.py` — `c88249ba2d4565710397491e8ead885ecdd0b2371c6875c25f20a807e30ed6ce`
- `Scripts/from_symile/constants.py` — `1bcddcf447ef4a42331a52e2d770cdeb71714c46e21b5fef66d2f2071e21bdf6`
- `Scripts/from_symile/datasets.py` — `75ad05fda814fe342fb57f44b1960b8aedf48ce9d90cd74d060823556c461c46`
- `Scripts/from_symile/env/environment.yml` — `1fbd90c6a5aff2c26e6c4d02a9484ce0ea3f2c55dae4ef21745189a4e4a15a33`
- `Scripts/from_symile/env/environment_mimic.yml` — `b437b958227d6476d591c9c9dd5dc7417fbb34197116de7a78aa23efcddb5c87`
- `Scripts/from_symile/env/requirements.txt` — `aa7cc8668633d8a04bf856cb0bfd19ac718dd4cd1285d5ebe8f01af62491c2ef`
- `Scripts/from_symile/losses.py` — `0c83032fe42d508fb7334e85c5d733926fa255d782baa867d754ca932a7831c6`
- `Scripts/from_symile/main.py` — `b698d644be6c2e65e49adcc672b3a88b7d575443b44db6cd86ead0eeeac273d7`
- `Scripts/from_symile/mmft_model_structure.txt` — `723c0d4fd7a194ae2a05537e29f72ff76d7e4a5071794ce6a9f6386d95d8d41b`
- `Scripts/from_symile/models/__init__.py` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `Scripts/from_symile/models/mmft_mimic_model.py` — `d17fc161c897d6656f88fdaddef20d68ae3ddd15caf938a49937bce4db80600b`
- `Scripts/from_symile/models/simsiam.py` — `1b55cb4e516e33acbc6e7e062b4b7b3b6db15356fc48e12da216051c06db4233`
- `Scripts/from_symile/models/stft_mimic_model.py` — `1ee00436c76bcdd83d3360fdf7c614d6cac8e88b6aa89f0e35abd5f92c612030`
- `Scripts/from_symile/models/symile_mimic_model.py` — `1863406c15944450b92f0f35d14fc3734f516f77fcdf6f8118cdeeb517464193`
- `Scripts/from_symile/preprocessing/__pycache__/args.cpython-311.pyc` — `a8bd23f01ada43df72e41b5b2a57dbcb891ec0f68fc2db0cb7400cc99d7511e6`
- `Scripts/from_symile/preprocessing/args.py` — `a38d727f84c4f1f1e7a0f25b39db171cac56412478f5192d26bc5f188af08afb`
- `Scripts/from_symile/preprocessing/constants.py` — `6e9b1d2361e8dcd19a520dfdcbee72d24e3e2566d692a78a7707bbf452f65547`
- `Scripts/from_symile/preprocessing/process_and_save_tensors.py` — `5787b64e56487da1e8ead6212d248d0d29c8ac8a4db8b5bf1988991537f3620d`
- `Scripts/from_symile/pyproject.toml` — `c0ad2637b8844b4923a93e799166cb02595b3d6a8175a61e51e4725f046f36cc`
- `Scripts/from_symile/test.py` — `88387ed83713dc2b7e8ae2f43921b286f8fa4db92f40deadd73800b78c266266`
- `Scripts/from_symile/utils.py` — `ae9bd3b9e8e6927ccfd9ebcbde0752dc5e9df83af493e9e3a8d2774cf1eaccfe`
- `Scripts/medmod/exclude_symile.py` — `ecfb8fe98f5248d00487d2fb28ee7682f042d0f9d795a8f07aeacf813bc598d2`
- `Scripts/medmod/link_mimic_cxr_jpg.py` — `77eea8df4772e119874cfa87266528f2eb50e261435e27678d16c2261525018a`
- `Scripts/medmod/query_mimic_cxr_jpg.py` — `16dd59cf34267fec38dd7523ab4ef98c5528aac247933ce3b3fe1ca406e9dd85`
- `Scripts/medmod/split_medmod.py` — `2d531cac0f84ce230c78a64709eec3dbd8ca6656fae7f12dbb398e092f8ac859`
- `Scripts/umap_vis.py` — `39c72090441c46b7401bf18be251a01321cc61636c57fb5e5be77971ec6cb440`
- `Stage1_MAE.py` — `ef632614822174d051926085700b2433cc4da3a56421d6b73f980dfcdff46f98`
- `Stage2_PACX.py` — `2823eaca30140f15ff32ce7c3a9005a8c6062bf5cac5c2486372974c74d7575c`
- `Utils/__init__.py` — `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `Utils/utils.py` — `db812111e39b07d32e3c81678ba68e827f13a2f51cc87639bbd37cbacbbd6083`
