import argparse
import os
import copy
import yaml
import cv2

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as CAWR
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from easydict import EasyDict as edict

from models.srcnn import SRCNN
from models.UNet import UNet
from models.drrn import DRRN

from lib.datasets import train_set, val_set
from lib.metrics import calc_psnr
from utils.meters import AverageMeter

cudnn.benchmark = True

model_dict = {
    'UNet': UNet,
    'DRRN': DRRN,
    'SRCNN': SRCNN
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='DRRN')
    parser.add_argument('--down_sample', type=str, default='delaunay')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--sample_method', type=str, default='FFT')
    parser.add_argument('--scale', type=int, default=3)

    args = parser.parse_args()
    return args


def get_config(args):
    cfg_path = args.config_path
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    return edict(config)


def set_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def save_img(input, name, args):
    input_tensor = input.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(
        0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(
        args.outputs_dir, '{}'.format(name)), input_tensor)


def load_dataset(args, cfg):
    train = train_set(args, cfg)
    val = val_set(args, cfg)
    train_loader = DataLoader(
        dataset=train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader, train, val


def eval_epoch(model, val_loader, device, args, bst):
    res = {}
    model.eval()
    input_psnr = AverageMeter()
    pred_psnr = AverageMeter()
    for i, (input, label) in tqdm(enumerate(val_loader), leave=False):
        input = input.to(device).float()
        label = label.to(device).float()
        res = label-input
        with torch.no_grad():
            pred = model(input)
            pred = pred.clamp(0.0, 1.0)
            if i == 9:
                save_img(input, 'input_{}.png'.format(i), args)
                save_img(pred, 'pred_{}.png'.format(i), args)
                save_img(label, 'gt_{}.png'.format(i), args)
                save_img(pred-input, 'pred_res_{}.png'.format(i), args)
                save_img(res, 'gt_res_{}.png'.format(i), args)
        input_psnr.update(calc_psnr(input, label), len(input))
        pred_psnr.update(calc_psnr(pred, label), len(input))

    print("input psnr: {:.2f}, pred psnr: {:.2f}".format(
        input_psnr.avg, pred_psnr.avg))
    if pred_psnr.avg > bst:
        bst = pred_psnr.avg
        best_weights = copy.deepcopy(model.state_dict())
        print('best psnr: {:.2f}'.format(bst))
        torch.save(best_weights, os.path.join(
            args.outputs_dir, 'best.pth'))
    return bst


def train_epoch(model, train, train_loader, device, epoch, args, criterion, optimizer, scheduler):
    model.train()
    epoch_loss = AverageMeter()

    with tqdm(total=(len(train) - len(train) % args.batch_size),
              leave=False) as t:
        t.set_description(
            'epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for input, label in train_loader:
            input = input.to(device).float()
            label = label.to(device).float()
            pred = model(input)

            loss = criterion(pred, label)
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10, norm_type=2)
            epoch_loss.update(loss.item(), len(input))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg), lr='{:.2e}'.format(
                optimizer.state_dict()['param_groups'][0]['lr']))
            t.update(len(input))


def main():
    args = parse_args()
    cfg = get_config(args=args)
    # set exp dir
    args.outputs_dir = os.path.join(
        args.outputs_dir, args.sample_method, '{}'.format(args.model))
    args.outputs_dir = os.path.join(
        args.outputs_dir, 'x_{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    # set device, seed
    device = set_device()
    torch.manual_seed(args.seed)
    # load dataset
    train_loader, val_loader, train, val = load_dataset(args, cfg)
    # set model
    model = model_dict[str(args.model)]().to(device)
    if not args.ckpt == None:
        print('loading checkpoint from {}'.format(args.ckpt))
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CAWR(optimizer, T_0=2*len(train),
                     T_mult=2, eta_min=0.1*args.lr)
    # set loss function
    criterion = nn.MSELoss()

    bst = 0.0
    for epoch in range(args.num_epochs):
        bst = eval_epoch(model, val_loader, device, args, bst)
        train_epoch(model, train, train_loader, device, epoch,
                    args, criterion, optimizer, scheduler)


if __name__ == "__main__":
    main()
