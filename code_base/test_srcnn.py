import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from torch.serialization import load

from models.srcnn import SRCNN
from utils.convert import rgb2y, rgb2ycbcr, ycbcr2rgb
from lib.metrics import calc_psnr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--mask_prob', type=float, default=0.0)
    args = parser.parse_args()
    return args


def set_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(model, args):
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    return model


def get_mask(input, prob=0.0):
    mask = np.random.choice(2, size=(input.shape[0], input.shape[1], 3),
                            p=[prob, 1-prob])
    return mask


def load_image(args, device):
    gt = pil_image.open(args.image_file).convert('RGB')
    # BICUBIC
    width = (gt.width // args.scale) * args.scale
    height = (gt.height // args.scale) * args.scale
    input = gt.resize((width, height), resample=pil_image.BICUBIC)
    input = input.resize((input.width//args.scale,
                          input.height//args.scale), resample=pil_image.BICUBIC)
    input = input.resize((input.width*args.scale,
                          input.height*args.scale), resample=pil_image.BICUBIC)
    input_np = np.array(input).astype(np.float32)
    # MASK
    mask = get_mask(input_np, args.mask_prob)
    masked_input = (input_np*mask).astype(np.float32)
    # rgb2ycbcr
    ycbcr = rgb2ycbcr(input_np)
    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    ycbcr_masked = rgb2ycbcr(masked_input)
    y_masked = ycbcr_masked[..., 0]
    y_masked /= 255.
    y_masked = torch.from_numpy(y_masked).to(device)
    y_masked = y_masked.unsqueeze(0).unsqueeze(0)
    return gt, input, y_masked, y, ycbcr, ycbcr_masked


def main():
    args = parse_args()
    device = set_device()
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    model = SRCNN().to(device)
    model = load_model(model, args)
    model.eval()
    gt, input, y_masked, y, ycbcr, ycbcr_masked = load_image(args, device)
    gt.save(os.path.join(args.outputs_dir, 'gt.bmp'))
    input.save(os.path.join(args.outputs_dir, 'bicubic.bmp'))
    with torch.no_grad():
        preds = model(y_masked).clamp(0.0, 1.0)
    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))
    
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    y = y.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    y_masked = y_masked.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    
    input = np.array([y_masked, ycbcr_masked[..., 1], ycbcr_masked[..., 2]]
                      ).transpose([1, 2, 0])
    input = np.clip(ycbcr2rgb(input), 0.0, 255.0).astype(np.uint8)
    input = pil_image.fromarray(input)
    input.save(os.path.join(args.outputs_dir, 'masked_input.bmp'))
    
    output = np.array([preds, ycbcr_masked[..., 1], ycbcr_masked[..., 2]]
                      ).transpose([1, 2, 0])
    output = np.clip(ycbcr2rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(os.path.join(args.outputs_dir, 'output.bmp'))


if __name__ == "__main__":
    main()
