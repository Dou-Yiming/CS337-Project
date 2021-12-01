import argparse
import os
import copy
import yaml

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from easydict import EasyDict as edict

from models.srcnn import SRCNN
from lib.datasets import SRCNN_train_set, SRCNN_eval_set
from lib.metrics import calc_psnr
from utils.meters import AverageMeter

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def get_config(args):
    cfg_path = args.config_path
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    return edict(config)


def set_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    cfg = get_config(args=args)

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    device = set_device()
    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW([
        {'params': model.conv.parameters()},
        {'params': model.head.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_set = SRCNN_train_set(args.train_file)
    eval_set = SRCNN_eval_set(args.eval_file)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    eval_loader = DataLoader(
        dataset=eval_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print("Dataset loaded, train: {}, eval: {}".format(
        len(train_loader), len(eval_loader)
    ))

    bst = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = AverageMeter()

        with tqdm(total=(len(train_set) - len(train_set) % args.batch_size),
                  leave=False) as t:
            t.set_description(
                'epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for input, label in train_loader:
                input = input.to(device).float()
                label = label.to(device).float()

                pred = model(input)

                loss = criterion(pred, label)
                epoch_loss.update(loss.item(), len(input))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg))
                t.update(len(input))

        model.eval()
        epoch_psnr = AverageMeter()

        for input, label in tqdm(eval_loader, leave=False):
            input = input.to(device).float()
            label = label.to(device).float()
            with torch.no_grad():
                pred = model(input).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(pred, label), len(input))

        if epoch_psnr.avg > bst:
            bst = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
            print('best psnr: {:.2f}'.format(bst))
            torch.save(best_weights, os.path.join(
                args.outputs_dir, 'best.pth'))


if __name__ == "__main__":
    main()
