import argparse
import time
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import torch

from Data.data_modules import CXRDataModule
from Modules.mae_lit import MAELightningModule



def main(args):
    
    current_time = time.strftime("%Y%m%d_%H%M%S") #used to give each run a unique name

    run_name = (
        f"mae_cxr_base"
        f"_bs{args.batch_size}"
        f"_mr{args.mask_ratio}"
        f"_lr{args.lr}"
        f"_wd{args.weight_decay}"
        f"_ep{args.max_epochs}"
        f"_we{args.warmup_epochs}" #the initial warmup epochs for the learning rate scheduler
        f"_{current_time}"
    )

    # root dir for ALL outputs of this run
    base_dir = Path(args.output_dir).expanduser().resolve()
    run_dir = base_dir / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
