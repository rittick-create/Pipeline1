"""Train the ECG + lab student to match a frozen Stage 1 MAE CXR teacher.

By default this entry point requires paired real data and all three encoders'
checkpoints. Synthetic data plus random weights are opt-in for software checks.
"""
import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from Data.stage2_data import Stage2DataModule
from Modules.reverse_pacx_lit import ReversePACXDistillation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--metadata-dir', type=Path)
    parser.add_argument('--cxr-checkpoint', type=Path)
    parser.add_argument('--ecg-checkpoint', type=Path)
    parser.add_argument('--labs-checkpoint', type=Path)
    parser.add_argument('--save-dir', type=Path, default=Path('checkpoints/stage2_reverse'))
    parser.add_argument('--resume-from', type=Path, default=None,
                        help='Resume model, optimizer, scheduler, and epoch from a Lightning checkpoint')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--clip-weight', type=float, default=0.5)
    parser.add_argument('--regression-weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--accelerator', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', default='32-true')
    parser.add_argument('--cxr-normalization', choices=['imagenet', 'zero_one'], default='imagenet')
    parser.add_argument('--ecg-lead-order', choices=['pacx_original', 'standard'],
                        default='pacx_original')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic inputs; results have no research meaning.')
    parser.add_argument('--allow-random-init', action='store_true',
                        help='Allow random encoder weights for implementation checks only.')
    args = parser.parse_args()
    if args.synthetic != args.allow_random_init:
        parser.error('--synthetic and --allow-random-init must be supplied together')
    if args.synthetic:
        if any((args.data_dir, args.metadata_dir, args.cxr_checkpoint,
                args.ecg_checkpoint, args.labs_checkpoint)):
            parser.error('Synthetic smoke checks do not accept data or checkpoint paths')
    else:
        missing = [name for name in ('data_dir', 'metadata_dir', 'cxr_checkpoint',
                                     'ecg_checkpoint', 'labs_checkpoint')
                   if getattr(args, name) is None]
        if missing:
            parser.error('Real training requires: ' + ', '.join('--' + n.replace('_', '-') for n in missing))
    if args.devices < 1:
        parser.error('--devices must be at least 1')
    return args


def main():
    args = parse_args()
    pl.seed_everything(42, workers=True)
    data = Stage2DataModule(
        data_dir=args.data_dir, metadata_dir=args.metadata_dir,
        batch_size=args.batch_size, num_workers=args.num_workers,
        synthetic=args.synthetic, cxr_normalization=args.cxr_normalization,
        ecg_lead_order=args.ecg_lead_order,
    )
    model = ReversePACXDistillation(
        clip_weight=args.clip_weight, regression_weight=args.regression_weight,
        temperature=args.temperature, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs,
    )
    if not args.synthetic:
        model.initialize_encoders(args.cxr_checkpoint, args.ecg_checkpoint,
                                  args.labs_checkpoint)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(dirpath=args.save_dir, filename='reverse-{epoch:03d}-{val/loss_total:.4f}',
                        monitor='val/loss_total', mode='min', save_top_k=1,
                        save_last=True, auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=args.devices, max_epochs=args.max_epochs,
        precision=args.precision, logger=CSVLogger(save_dir=str(args.save_dir), name='logs'),
        callbacks=callbacks, default_root_dir=str(args.save_dir),
        gradient_clip_val=1.0, log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data,
                ckpt_path=str(args.resume_from) if args.resume_from else None)
    if trainer.is_global_zero:
        print('Best checkpoint:', callbacks[0].best_model_path)
        print('Encoder sources:', model.encoder_sources)


if __name__ == '__main__':
    main()
