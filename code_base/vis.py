import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
# input_500 = plt.imread("./exp/UNet/delaunay/500/input_0.png")
# pred_500 = plt.imread("./exp/UNet/delaunay/500/pred_0.png")
# input_2000 = plt.imread("./exp/UNet/delaunay/2000/input_0.png")
# pred_2000 = plt.imread("./exp/UNet/delaunay/2000/pred_0.png")
# input_10000 = plt.imread("./exp/UNet/delaunay/10000/input_0.png")
# pred_10000 = plt.imread("./exp/UNet/delaunay/10000/pred_0.png")
# input_20000 = plt.imread("./exp/UNet/delaunay/20000/input_0.png")
# pred_20000 = plt.imread("./exp/UNet/delaunay/20000/pred_0.png")
# plt.subplot(241), plt.imshow(input_500), plt.title(
#         "input-500"), plt.xticks([]), plt.yticks([])
# plt.subplot(245), plt.imshow(pred_500), plt.title(
#         "pred-500"), plt.xticks([]), plt.yticks([])

# plt.subplot(242), plt.imshow(input_2000), plt.title(
#         "input-2000"), plt.xticks([]), plt.yticks([])
# plt.subplot(246), plt.imshow(pred_2000), plt.title(
#         "pred-2000"), plt.xticks([]), plt.yticks([])

# plt.subplot(243), plt.imshow(input_10000), plt.title(
#         "input-10000"), plt.xticks([]), plt.yticks([])
# plt.subplot(247), plt.imshow(pred_10000), plt.title(
#         "pred-10000"), plt.xticks([]), plt.yticks([])

# plt.subplot(244), plt.imshow(input_20000), plt.title(
#         "input-20000"), plt.xticks([]), plt.yticks([])
# plt.subplot(248), plt.imshow(pred_20000), plt.title(
#         "pred-20000"), plt.xticks([]), plt.yticks([])
# plt.show()


input_1 = plt.imread("./MAE-pytorch/output/random/mask_img_0.1.jpg")
pred_1 = plt.imread("./MAE-pytorch/output/random/rec_img_0.1.jpg")
input_2 = plt.imread("./MAE-pytorch/output/random/mask_img_0.2.jpg")
pred_2 = plt.imread("./MAE-pytorch/output/random/rec_img_0.2.jpg")
input_4 = plt.imread("./MAE-pytorch/output/random/mask_img_0.4.jpg")
pred_4 = plt.imread("./MAE-pytorch/output/random/rec_img_0.4.jpg")
input_8 = plt.imread("./MAE-pytorch/output/random/mask_img_0.8.jpg")
pred_8 = plt.imread("./MAE-pytorch/output/random/rec_img_0.8.jpg")
plt.subplot(241), plt.imshow(input_1), plt.title(
        "input (10% masked)"), plt.xticks([]), plt.yticks([])
plt.subplot(245), plt.imshow(pred_1), plt.title(
        "rec (10% masked)"), plt.xticks([]), plt.yticks([])

plt.subplot(242), plt.imshow(input_2), plt.title(
        "input (20% masked)"), plt.xticks([]), plt.yticks([])
plt.subplot(246), plt.imshow(pred_2), plt.title(
        "rec (20% masked)"), plt.xticks([]), plt.yticks([])

plt.subplot(243), plt.imshow(input_4), plt.title(
        "input (40% masked)"), plt.xticks([]), plt.yticks([])
plt.subplot(247), plt.imshow(pred_4), plt.title(
        "rec (40% masked)"), plt.xticks([]), plt.yticks([])

plt.subplot(244), plt.imshow(input_8), plt.title(
        "input (80% masked)"), plt.xticks([]), plt.yticks([])
plt.subplot(248), plt.imshow(pred_8), plt.title(
        "rec (80% masked)"), plt.xticks([]), plt.yticks([])
plt.show()