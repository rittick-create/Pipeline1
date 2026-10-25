import argparse
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from models import MainTrainer
from utils import create_dataloaders


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the CXR, ECG, and lab model for 14 CheXpert labels.")
    parser.add_argument("--data_dir", type=Path,
                        default=project_dir.parent / "symile-mimic" / "symile-mimic-aws")
    parser.add_argument("--output_dir", type=Path, default=project_dir / "runs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cxr_missing_ratio", type=float, default=0.30)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--ecg_n_layers", type=int, default=1)
    parser.add_argument("--ecg_n_head", type=int, default=8)
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
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    train_dl, val_dl, test_dl = create_dataloaders(
        dataset_root=args.data_dir,
        batch_size=args.batch_size,
        train_missing_ratio=args.cxr_missing_ratio,
        val_missing_ratio=args.cxr_missing_ratio,
        test_missing_ratio=args.cxr_missing_ratio,
        num_workers=0,
        seed=args.seed,
    )
    model = MainTrainer(args=args, label_names=train_dl.dataset.CLASSES)

    # checkpoint with the highest mean validation average precision.
    monitor = "val_PRAUC_avg_over_dxs/final"
    checkpoint = ModelCheckpoint(monitor=monitor, mode="max",
                                 save_top_k=1, filename="best")
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

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.test(model=model, dataloaders=test_dl, ckpt_path="best")

    # Keep predictions, masks, features, and attention weights for later analysis.
    results = {
        "args": vars(args),
        "best_checkpoint": checkpoint.best_model_path,
        "best_val_prauc": checkpoint.best_model_score.item(),
        "test_results": model.test_results,
    }
    result_path = Path(logger.log_dir) / "test_results.pt"
    torch.save(results, result_path)
    print("Best checkpoint:", checkpoint.best_model_path)
    print("Test results:", result_path)
