# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

from filters import high_pass_filtering
from einops import rearrange


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self, img=None):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196]


class FFTMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self, img):
        img = img.detach().clone()
        img = img.permute([1, 2, 0]).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = high_pass_filtering(img, 10, 1)
        img = np.expand_dims(np.expand_dims(img, axis=2), axis=0)
        img = rearrange(
            img, 'b (h p_h) (w p_w) c -> b (h w) (p_h p_w) c', p_h=16, p_w=16)
        img = np.mean(img, axis=2).squeeze(2).squeeze(0)
        img = np.argsort(img)

        mask = np.zeros(self.num_patches)
        mask[img[:self.num_mask]] = 1

        return mask  # [196]
