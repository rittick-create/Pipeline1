"""PACX-MAE Stage 1 CLI adapted for the local CheXpert layout and run settings."""

from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from Data.data_modules import CXRDataModule
from Utils.runner import parse_args as stage_args, make_model, validate_checkpoint


def parse_args(argv=None):
    return stage_args("mae", argv)


def main(args):
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    validate_checkpoint("mae", args)

    data_module = CXRDataModule(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        sample_limit=args.sample_limit,
    )
    # This builds the original MAELightningModule and records the run settings.
    model, monitor, mode = make_model("mae", args)
    logger = CSVLogger(save_dir=str(args.output_dir), name="")
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        filename="best",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb, LearningRateMonitor(logging_interval="epoch")],
        deterministic=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        default_root_dir=str(args.output_dir),
    )
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=str(args.resume) if args.resume else None,
    )
    if not checkpoint_cb.best_model_path:
        raise RuntimeError("No best checkpoint was saved; inspect validation loss")
    print("Run directory:", Path(logger.log_dir))
    print("Best checkpoint:", checkpoint_cb.best_model_path)


if __name__ == "__main__":
    main(parse_args())
