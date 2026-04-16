
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from predictors import QCNet

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--limit_train_batches', type=float, default=1.,
                        help='训练集抽样比例(0-1)或批次数量(>1)')
    parser.add_argument('--limit_val_batches', type=float, default=1.,
                        help='验证集抽样比例(0-1)或批次数量(>1)')
    parser.add_argument('--limit_test_batches', type=float, default=1.,
                        help='测试集抽样比例(0-1)或批次数量(>1)')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--ckpt_path', type=str, default=None)
    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = QCNet(**vars(args))
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches, 
                         limit_test_batches=args.limit_test_batches,
                         precision='16-mixed')
    trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
