import os
import numpy as np
import torch
import random
import torch.utils.data as data
from os import path as osp
import scipy.io as sio
import kornia
from PIL import Image
from kornia.augmentation import RandomCrop


class Photosketch_Kornia_Set(data.Dataset):
    affine_type: object

    def __init__(self, opt, forTrain=True, use_par=True, ske_to_img=False, additional_bright=False):
        super(Photosketch_Kornia_Set, self).__init__()
        self.root = opt.root
        self.imgResol = opt.input_size
        self.ske_to_img = ske_to_img
        if forTrain:
            self.listfile = osp.join(opt.root, opt.dataset_ref, opt.dataset_train_list)
        else:
            self.listfile = osp.join(opt.root, opt.dataset_ref, opt.dataset_test_list)
        self.add_bright = additional_bright
        self.forTrain = forTrain
        self.lens = self.getLines()
        self.opt = opt
        self.type = opt.dataset_name
        self.use_par = use_par
        self.affine_type = opt.affine_type  # 对训练集做大型偏移的类型
        self.img_nc = opt.image_nc
        self.par_nc = opt.parsing_nc
        self.tar_nc = opt.output_nc
        self.loaded_pool = {}  # 这里dataloader 不要开num_of_worker

    def __getitem__(self, index):
        line = self.lines[index]
        try:
            tensors = self.loaded_pool[index]
        except KeyError:
            tensors = self.loadImg(line)
            self.loaded_pool[index] = tensors
        return tensors

    def __len__(self):
        return len(self.lines)

    def apply_tranform(self, tensors):

        if self.forTrain:  # apply same Flip in source and target while training
            tensors = kornia.augmentation.RandomHorizontalFlip()(tensors)
        if len(tensors.size()) == 3:  # bsize = 1
            tensors = tensors.unsqueeze(0)

        # for x and y transform methods
        x_trans_arr = []
        y_trans_arr = []
        # share transform
        y_trans_arr.append(kornia.geometry.Resize((self.opt.img_h, self.opt.img_w)))
        if self.forTrain:
            loadSize = int(np.ceil(self.opt.input_size * 1.117))
            if loadSize % 2 == 1:
                loadSize += 1
            if self.affine_type == "normal":
                y_trans_arr.append(kornia.augmentation.CenterCrop((loadSize, loadSize)))
            elif self.affine_type == "width":
                y_trans_arr.append(RandomCrop((loadSize, loadSize), pad_if_needed=True))
            elif self.affine_type == "scale":
                scale_factor = random.gauss(1, 0.15)
                if scale_factor > 1.3:
                    scale_factor = 1.3
                elif scale_factor < 0.7:
                    scale_factor = 0.7
                n_w = round(self.opt.img_w * scale_factor)
                n_h = round(self.opt.img_h * scale_factor)

                y_trans_arr.append(kornia.Resize((n_h, n_w)))
                y_trans_arr.append(kornia.augmentation.CenterCrop((loadSize, loadSize)))

            y_trans_arr.append(RandomCrop(size=(self.opt.input_size, self.opt.input_size), pad_if_needed=True))

            x_trans_arr.append(kornia.augmentation.CenterCrop((loadSize, loadSize)))
            x_trans_arr.append(RandomCrop(size=(self.opt.input_size, self.opt.input_size), pad_if_needed=True))
        else:  # test
            y_trans_arr.append(kornia.augmentation.CenterCrop((self.opt.input_size, self.opt.input_size)))
            x_trans_arr.append(kornia.augmentation.CenterCrop((self.opt.input_size, self.opt.input_size)))

        y_trans_method = torch.nn.Sequential(*y_trans_arr)
        y_trans_method = y_trans_method
        # split
        org_nc = self.img_nc
        if self.use_par:
            org_nc += self.par_nc

        if self.affine_type == "normal":
            tensors = y_trans_method(tensors)
        else:  # 随机变化 放大x y 差距
            x_trans_method = torch.nn.Sequential(*x_trans_arr)
            x_trans_method = x_trans_method
            src = tensors[:, :org_nc]
            tar = tensors[:, org_nc:]
            tar = y_trans_method(tar)
            src = x_trans_method(src)
            tensors = torch.cat([src, tar], dim=1)

        # normalized
        src_img = tensors[:, :self.img_nc]
        src_par = tensors[:, self.img_nc:org_nc]
        tar_img = tensors[:, org_nc:org_nc + self.tar_nc]
        tar_par = tensors[:, org_nc + self.tar_nc:]
        src_img = kornia.enhance.Normalize(0.5, 0.5)(src_img)
        tar_img = kornia.enhance.Normalize(0.5, 0.5)(tar_img)

        src = torch.cat([src_img, src_par], dim=1)
        tar = torch.cat([tar_img, tar_par], dim=1)

        if self.ske_to_img:
            tmp = src
            src = tar
            tar = tmp
        return src, tar

    def getLines(self):
        self.lines = []
        with open(self.listfile, 'r') as f:  # infofile
            for line in f:
                line = line.strip()
                self.lines.append(line)
        lens = len(self.lines)
        return lens

    def get_mat(self, matpath):
        facelabel = sio.loadmat(matpath)
        temp = facelabel['res_label']
        return temp

    def loadImg(self, line):
        items = line.split('||')
        inPath1 = os.path.join(self.root, items[0])
        src = Image.open(inPath1).convert("RGB")
        inPath2 = os.path.join(self.root, items[1])
        tar = Image.open(inPath2)
        # read pic
        src = kornia.image_to_tensor(np.array(src, dtype=float)).float() / 255
        tar = kornia.image_to_tensor(np.array(tar, dtype=float)).float()
        if tar.size(0) == 3:
            tar = kornia.color.RgbToGrayscale()(tar)
        tar = tar / 255

        # merge mat
        matPath1 = os.path.join(self.root, items[2])
        matPath2 = os.path.join(self.root, items[3])
        if self.use_par:
            par_mat = self.get_mat(matPath1)
            ske_mat = self.get_mat(matPath2)
            par_mat = kornia.image_to_tensor(par_mat, keepdim=False).float()
            ske_mat = kornia.image_to_tensor(ske_mat, keepdim=False).float()
            gauss = kornia.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(0.2, 0.2))
            par_mat = gauss(par_mat)[0]
            ske_mat = gauss(ske_mat)[0]
            src = torch.cat([src, par_mat])
            tar = torch.cat([tar, ske_mat])
        return torch.cat([src, tar])  # 合在一起操作，后再将其分开
