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
from torchvision import transforms
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='DRRN')
    parser.add_argument('--ckpt', type=str, default=None)
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


def main():
    args = parse_args()
    # set exp dir
    args.outputs_dir = os.path.join(
        args.outputs_dir, '{}'.format(args.model))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    # set device, seed
    device = set_device()
    torch.manual_seed(args.seed)
    # load data
    img_path = './data/tmp/10_1.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tran = transforms.ToTensor()
    img = tran(img).unsqueeze(0)
    img = img.to(device).float()
    # set model
    model = model_dict[str(args.model)]().to(device)
    if not args.ckpt == None:
        print('loading checkpoint from {}'.format(args.ckpt))
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
    pred = model(img)
    save_img(pred, 'pred.png', args)


if __name__ == "__main__":
    main()
