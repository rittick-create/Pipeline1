import argparse
from pathlib import Path

import torch
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from models import MainTrainer
from utils import TARGETS, create_dataloaders, create_test_dataloaders
from utils.evaluation import save_comparison


TRAINING_PROTOCOL = "ecgfounder_complete_inputs_three_tests_v1"


def parse_args(argv=None):
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the CXR, ECG, and lab model for 14 CheXpert labels.")
    parser.add_argument("--data_dir", type=Path,
                        default=project_dir.parent / "symile-mimic" / "symile-mimic-aws")
    parser.add_argument("--output_dir", type=Path, default=project_dir / "runs_complete")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--resume", type=Path, default=None,
                             help="Resume a run from this complete-input protocol.")
    checkpoints.add_argument("--eval_checkpoint", type=Path, default=None,
                             help="Skip training and evaluate a complete-input checkpoint three ways.")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--ecg_checkpoint", type=Path,
                        default=project_dir.parent / "ecgfounder" / "checkpoint" / "12_lead_ECGFounder.pth",
                        help="Official pretrained 12-lead ECGFounder weights.")
    parser.add_argument("--freeze_ecg", action="store_true",
                        help="Freeze the pretrained backbone; train its new branches and the rest of the model.")
    parser.add_argument("--ecg_lr", type=float, default=1e-5,
                        help="Learning rate for the pretrained ECG backbone.")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--lambda_disentangle_shared", type=float, default=1)
    parser.add_argument("--lambda_disentangle_ecg", type=float, default=1)
    parser.add_argument("--lambda_disentangle_labs", type=float, default=1)
    parser.add_argument("--lambda_disentangle_cxr", type=float, default=1)
    parser.add_argument("--lambda_pred_shared", type=float, default=1)
    parser.add_argument("--lambda_pred_ecg", type=float, default=1)
    parser.add_argument("--lambda_pred_labs", type=float, default=1)
    parser.add_argument("--lambda_pred_cxr", type=float, default=1)
    parser.add_argument("--lambda_attn_aux", type=float, default=1)
    args = parser.parse_args(argv)
    args.training_protocol = TRAINING_PROTOCOL
    return args


def validate_checkpoint(path, args):
    # Only use trusted local checkpoints. mmap avoids copying optimizer tensors.
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    hparams = saved["hyper_parameters"]
    if hparams.get("training_protocol") != TRAINING_PROTOCOL:
        raise ValueError("This checkpoint used a different training protocol. Start a fresh run.")
    if list(hparams.get("label_names", [])) != TARGETS:
        raise ValueError("The checkpoint's prediction targets do not match TARGETS.")
    settings = ["hidden_size", "dropout", "freeze_ecg", "ecg_lr",
                "lr", "wd", "seed", "patience"]
    settings += [name for name in vars(args) if name.startswith("lambda_")]
    for name in settings:
        if hparams[name] != getattr(args, name):
            raise ValueError(f"Use --{name} {hparams[name]} to match the checkpoint.")


def main(args):
    if args.resume is not None or args.eval_checkpoint is not None:
        validate_checkpoint(args.resume or args.eval_checkpoint, args)

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if args.eval_checkpoint is None:
        train_dl, val_dl, test_dls = create_dataloaders(
            args.data_dir, batch_size=args.batch_size, seed=args.seed,
        )
        datasets = {"train": train_dl.dataset, "val": val_dl.dataset,
                    "test": test_dls["all"].dataset}
    else:
        test_dls = create_test_dataloaders(
            args.data_dir, batch_size=args.batch_size, seed=args.seed,
        )
        datasets = {"test": test_dls["all"].dataset}
    model = MainTrainer(args=args, label_names=TARGETS)

    # checkpoint with the highest mean validation average precision.
    monitor = "val_AP_avg_over_dxs/final"
    checkpoint = ModelCheckpoint(monitor=monitor, mode="max",
                                 save_top_k=1, save_last=True, filename="best")
    early_stop = EarlyStopping(monitor=monitor, mode="max", patience=args.patience)
    logger = CSVLogger(save_dir=str(args.output_dir), name="")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint, early_stop],
        logger=logger,
        log_every_n_steps=1,
    )

    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    label_counts = pd.concat([
        dataset.label_counts().assign(split=split)
        for split, dataset in datasets.items()
    ], ignore_index=True)
    label_counts.to_csv(run_dir / "label_counts.csv", index=False)

    if args.eval_checkpoint is None:
        trainer.fit(
            model=model, train_dataloaders=train_dl, val_dataloaders=val_dl,
            ckpt_path=str(args.resume) if args.resume is not None else None,
        )
        best_path = checkpoint.best_model_path
        if not best_path:
            raise RuntimeError("No best checkpoint was saved; check validation macro AP.")
    else:
        best_path = str(args.eval_checkpoint.resolve())

    # Reload the exact same frozen checkpoint for each condition.
    test_results = {}
    for condition, test_dl in test_dls.items():
        trainer.test(model=model, dataloaders=test_dl, ckpt_path=best_path)
        result = model.test_results
        result["condition"] = condition
        result["label_counts"] = test_dl.dataset.label_counts().to_dict("records")
        test_results[condition] = result
        torch.save({
            "args": vars(args), "best_checkpoint": best_path,
            "test_results": result,
        }, run_dir / f"test_results_{condition}.pt")

    # Keep predictions, masks, features, and attention weights for later analysis.
    summary = save_comparison(test_results, test_dls["all"].dataset.label_counts(), run_dir)
    print("Best checkpoint:", best_path)
    print(summary.to_string(index=False))
    print("Results and label counts:", run_dir)


if __name__ == "__main__":
    main(parse_args())
