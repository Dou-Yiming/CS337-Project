import torch


def calc_psnr(img1, img2):
    # return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * torch.log10(1. / torch.sqrt(mse))
