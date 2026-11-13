"""Shared CLI and local Lightning orchestration for the four stage entry points."""

import argparse
from pathlib import Path
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from Data.data_modules import PipelineDataModule
from Data.targets import TARGETS
from Models.models import DEFAULT_ECG, read_checkpoint
from Modules.mae_lit import MAELightningModule
from Modules.labs_lit import LabsDAELightning
from Modules.pacx_lit import CrossModalDistillation
from Modules.cl_lit import ClassificationLightningModule

PROTOCOL = "pipeline1_reverse_independent_students_v1"
PROJECT = Path(__file__).resolve().parents[1]


def parse_args(stage, argv=None):
    parser = argparse.ArgumentParser(description=f"Pipeline1 {stage}")
    default_root = (
        PROJECT.parent / "chexpert/chexpertchestxrays-u20210408"
        if stage == "mae"
        else PROJECT.parent / "symile-mimic/symile-mimic-aws"
    )
    parser.add_argument("--data_dir", type=Path, default=default_root)
    parser.add_argument("--output_dir", type=Path, default=PROJECT / "runs" / stage)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--epochs",
        type=int,
        default={"mae": 400, "labs": 50, "distillation": 50, "classification": 40}[
            stage
        ],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default={
            "mae": 1.5e-4,
            "labs": 3e-4,
            "distillation": 1e-4,
            "classification": 3e-3,
        }[stage],
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default={
            "mae": 0.05,
            "labs": 0.01,
            "distillation": 0.05,
            "classification": 0.01,
        }[stage],
    )
    parser.add_argument("--mask_ratio", type=float, default=0.90)
    parser.add_argument("--clip_weight", type=float, default=0.5)
    parser.add_argument("--regression_weight", type=float, default=1.0)
    parser.add_argument("--cxr_checkpoint", type=Path)
    parser.add_argument("--labs_checkpoint", type=Path)
    parser.add_argument("--ecg_checkpoint", type=Path, default=DEFAULT_ECG)
    parser.add_argument("--stage2_checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--accelerator", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument(
        "--sample_limit",
        type=int,
        help="Limit each cohort for an implementation check; leave unset for research runs.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--eval_checkpoint", type=Path, help="Stage 3 evaluation only")
    args = parser.parse_args(argv)
    if (
        args.batch_size < 1
        or args.epochs < 1
        or (stage == "distillation" and args.batch_size < 2)
    ):
        parser.error("Use positive epoch/batch counts; Stage 2 needs batch_size >= 2")
    if args.sample_limit is not None and args.sample_limit < args.batch_size:
        parser.error("sample_limit must cover at least one full batch")
    if not 0 < args.mask_ratio < 1:
        parser.error("mask_ratio must be between zero and one")
    if (
        min(args.clip_weight, args.regression_weight) < 0
        or args.clip_weight + args.regression_weight == 0
    ):
        parser.error("Loss weights must be nonnegative with at least one positive")
    if args.eval_checkpoint and stage != "classification":
        parser.error("Testing is reserved for Stage 3")
    if not args.resume and not args.eval_checkpoint:
        required = {
            "distillation": ["cxr_checkpoint", "labs_checkpoint"],
            "classification": ["stage2_checkpoint"],
        }.get(stage, [])
        for name in required:
            if getattr(args, name) is None:
                parser.error(f"--{name} is required")
    return args


def make_model(stage, args):
    if stage == "mae":
        model = MAELightningModule(
            size="base",
            mask_ratio=args.mask_ratio,
            lr=args.lr,
            weight_decay=args.weight_decay,
            log_images_every_n_epochs=0,
        )
        monitor, mode = "val/loss", "min"
    elif stage == "labs":
        model = LabsDAELightning(
            input_dim=50, lr=args.lr, weight_decay=args.weight_decay
        )
        monitor, mode = "val/denoising_mse", "min"
    elif stage == "distillation":
        model = CrossModalDistillation(
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            clip_weight=args.clip_weight,
            regression_weight=args.regression_weight,
        )
        if not args.resume:
            model.initialize_encoders(
                args.cxr_checkpoint, args.ecg_checkpoint, args.labs_checkpoint
            )
        monitor, mode = "val/loss_total", "min"
    else:
        model = ClassificationLightningModule(
            lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.epochs
        )
        if not args.resume and not args.eval_checkpoint:
            model.model.load_students(args.stage2_checkpoint)
        monitor, mode = "val/auroc_macro", "max"
    model.save_hyperparameters(
        {
            "stage": stage,
            "protocol": PROTOCOL,
            "seed": args.seed,
            "sample_limit": args.sample_limit,
            "data_dir": str(args.data_dir.resolve()),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "precision": args.precision,
            "targets": list(TARGETS) if stage == "classification" else [],
            "stage2_checkpoint": str(args.stage2_checkpoint),
        }
    )
    return model, monitor, mode


def validate_checkpoint(stage, args):
    source = args.resume or args.eval_checkpoint
    if source:
        saved = read_checkpoint(source)
        h = saved["hyper_parameters"]
        if h.get("stage") != stage or h.get("protocol") != PROTOCOL:
            raise ValueError("Checkpoint belongs to a different stage/protocol")
        for key in ("lr", "weight_decay", "seed", "sample_limit"):
            if h[key] != getattr(args, key):
                raise ValueError(
                    f"Checkpoint setting differs for {key}; use its saved value {h[key]}"
                )
        if stage == "mae" and h["mask_ratio"] != args.mask_ratio:
            raise ValueError("Resume with the saved mask ratio")
        if stage == "distillation":
            for key in ("clip_weight", "regression_weight"):
                if h[key] != getattr(args, key):
                    raise ValueError(f"Resume with saved {key}")
        if stage == "classification" and h.get("targets", TARGETS) != TARGETS:
            raise ValueError("Checkpoint targets or target order differ")
        del saved
    return source, h if source else {}


def run(stage, argv=None):
    args = parse_args(stage, argv)
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source, h = validate_checkpoint(stage, args)
    data = PipelineDataModule(
        stage,
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.seed,
        args.sample_limit,
    )
    model, monitor, mode = make_model(stage, args)
    if source:
        # Retain initialization provenance when resuming or evaluating.
        provenance = {
            key: h[key] for key in ("encoder_sources", "stage2_checkpoint") if key in h
        }
        model.save_hyperparameters(provenance)
    logger = CSVLogger(str(args.output_dir), name="")
    checkpoint = ModelCheckpoint(
        monitor=monitor, mode=mode, save_top_k=1, save_last=True, filename="best"
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor("epoch")],
        log_every_n_steps=1,
    )
    run_dir = Path(logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_checkpoint:
        best = str(args.eval_checkpoint.resolve())
    else:
        trainer.fit(
            model, datamodule=data, ckpt_path=str(args.resume) if args.resume else None
        )
        best = checkpoint.best_model_path
        if not best:
            raise RuntimeError("No best checkpoint saved; inspect validation metrics")
    print("Best checkpoint:", best)
    if stage == "classification":
        # This is the first point at which any stage can open SYMILE test data.
        trainer.test(model, datamodule=data, ckpt_path=best)
        result = model.test_results
        result["checkpoint"] = best
        result["inputs"] = ["ecg", "labs"]
        result["sample_limit"] = args.sample_limit
        torch.save(result, run_dir / "test_results_student.pt")
        counts = data.datasets["test"].label_counts(args.sample_limit)
        counts["AP"] = result["ap_per_finding"].numpy()
        counts["AUROC"] = result["auroc_per_finding"].numpy()
        counts.to_csv(run_dir / "test_per_finding.csv", index=False)
        pd.DataFrame(
            [
                {
                    "method": "ecg_labs_student",
                    "admissions": len(result["hadm_ids"]),
                    "macro_AP": result["macro_ap"],
                    "macro_AUROC": result["macro_auroc"],
                    "AP_findings": int(result["ap_per_finding"].isfinite().sum()),
                    "AUROC_findings": int(result["auroc_per_finding"].isfinite().sum()),
                    "sample_limit": args.sample_limit,
                    "data_dir": str(args.data_dir.resolve()),
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "precision": args.precision,
                    "targets": list(TARGETS) if stage == "classification" else [],
                    "checkpoint": best,
                }
            ]
        ).to_csv(run_dir / "test_summary.csv", index=False)
        for split, ds in data.datasets.items():
            ds.label_counts(args.sample_limit).assign(split=split).to_csv(
                run_dir / f"label_counts_{split}.csv", index=False
            )
    print("Run directory:", run_dir)
