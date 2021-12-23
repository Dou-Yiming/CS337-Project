import argparse
import os
import copy
import yaml
import cv2

import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from easydict import EasyDict as edict

from models.srcnn import SRCNN
from lib.datasets import train_set, val_set
from lib.metrics import calc_psnr
from utils.meters import AverageMeter
from models.UNet import UNet

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='UNet')
    parser.add_argument('--down_sample', type=str, default='delaunay')
    parser.add_argument('--point_num', type=int, default=10000)  # for delaunay
    parser.add_argument('--scale', type=int, default=3)  # for BICUBIC

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


def main():
    args = parse_args()
    cfg = get_config(args=args)

    args.outputs_dir = os.path.join(args.outputs_dir, '{}'.format(
        args.down_sample))
    args.outputs_dir = os.path.join(
        args.outputs_dir, 'x_{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    device = set_device()
    torch.manual_seed(args.seed)

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
    if args.model == 'UNet':
        model = UNet(input_channel=3, output_channel=3).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              weight_decay=1e-5, momentum=0.9)
    elif args.model == 'SRCNN':
        model = SRCNN(num_channels=3).to(device)
        optimizer = optim.SGD([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv_block.parameters()},
            {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr, weight_decay=1e-4, momentum=0.9)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=6080, T_mult=2, eta_min=1e-3
    )

    criterion = nn.MSELoss()

    bst = 0.0
    for epoch in range(args.num_epochs):
        model.eval()
        input_psnr = AverageMeter()
        pred_psnr = AverageMeter()
        for i, (input, label) in tqdm(enumerate(val_loader), leave=False):
            input = input.to(device).float()
            label = label.to(device).float()
            label -= input
            with torch.no_grad():
                pred = model(input)
                pred = pred.clamp(0.0, 1.0)
            input_psnr.update(calc_psnr(input, label+input), len(input))
            pred_psnr.update(calc_psnr(pred+input, label+input), len(input))
            save_img(input, 'input_{}.png'.format(i), args)
            save_img(pred+input, 'pred_{}.png'.format(i), args)
            save_img(label+input, 'gt_{}.png'.format(i), args)
            save_img(pred, 'pred_res_{}.png'.format(i), args)
            save_img(label, 'gt_res_{}.png'.format(i), args)

        print("input psnr: {:.2f}, pred psnr: {:.2f}".format(
            input_psnr.avg, pred_psnr.avg))
        if pred_psnr.avg > bst:
            bst = pred_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
            print('best psnr: {:.2f}'.format(bst))
            torch.save(best_weights, os.path.join(
                args.outputs_dir, 'best.pth'))
            save_img(input, 'input_{}.png'.format(i), args)
            save_img(pred, 'pred_{}.png'.format(i), args)
            save_img(label, 'gt_{}.png'.format(i), args)

        model.train()
        epoch_loss = AverageMeter()

        with tqdm(total=(len(train) - len(train) % args.batch_size),
                  leave=False) as t:
            t.set_description(
                'epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for input, label in train_loader:
                input = input.to(device).float()
                label = label.to(device).float()
                label -= input
                pred = model(input)

                loss = criterion(pred, label)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                epoch_loss.update(loss.item(), len(input))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_loss.avg), lr='{:.2e}'.format(
                    optimizer.state_dict()['param_groups'][0]['lr']))
                t.update(len(input))


if __name__ == "__main__":
    main()
