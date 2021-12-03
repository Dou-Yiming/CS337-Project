import pickle
import os.path as osp
import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from utils.filters import high_pass_filtering


class SRCNN_train_set(Dataset):
    def __init__(self, h5_file_path):
        super(SRCNN_train_set, self).__init__()
        self.h5_file_path = h5_file_path
        with h5py.File(self.h5_file_path, 'r') as f:
            self.inputs = list(f['lr'])
            self.labels = list(f['hr'])

    def __getitem__(self, index):
        input = np.expand_dims(self.inputs[index] / 255., 0)
        label = np.expand_dims(self.labels[index] / 255., 0)
        return input, label

    def __len__(self):
        return len(self.inputs)


class SRCNN_eval_set(Dataset):
    def __init__(self, h5_file_path):
        super(SRCNN_eval_set, self).__init__()
        self.h5_file_path = h5_file_path

    def __getitem__(self, index):
        with h5py.File(self.h5_file_path, 'r') as f:
            input = np.expand_dims(f['lr'][str(index)][:, :] / 255., 0)
            label = np.expand_dims(f['hr'][str(index)][:, :] / 255., 0)
        return input, label

    def __len__(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            return len(f['lr'])


class train_set(Dataset):
    def __init__(self, config):
        super(train_set, self).__init__()
        self.data_dir = config.TRAIN.DATA_DIR
        self.img_dir = config.TRAIN.IMG_DIR
        self.config = config
        self.db = pickle.load(open(osp.join(
            self.data_dir, self.config.TRAIN.DB
        ), 'rb'))

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        file_name = self.db[index]['file_name']
        gt = cv2.imread(osp.join(self.img_dir, file_name))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        height, width = gt.shape[0:2]
        # BICUBIC
        input = cv2.resize(gt, (width//self.config.TRAIN.SCALE,
                                height//self.config.TRAIN.SCALE),
                           interpolation=cv2.INTER_CUBIC)
        input = cv2.resize(input, (width, height),
                           interpolation=cv2.INTER_CUBIC)
        tran = transforms.ToTensor()
        return tran(input), tran(gt)


class val_set(Dataset):
    def __init__(self, config):
        super(val_set, self).__init__()
        self.data_dir = config.VAL.DATA_DIR
        self.img_dir = config.VAL.IMG_DIR
        self.config = config
        self.db = pickle.load(open(osp.join(
            self.data_dir, self.config.VAL.DB
        ), 'rb'))

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        file_name = self.db[index]['file_name']
        gt = cv2.imread(osp.join(self.img_dir, file_name))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        height, width = gt.shape[0:2]
        # BICUBIC
        input = cv2.resize(gt, (width//self.config.VAL.SCALE,
                                height//self.config.VAL.SCALE),
                           interpolation=cv2.INTER_CUBIC)
        input = cv2.resize(input, (width, height),
                           interpolation=cv2.INTER_CUBIC)
        tran = transforms.ToTensor()
        return tran(input), tran(gt)
