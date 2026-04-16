
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    test_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='test')
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    trainer.test(model, dataloader)
