# -*- coding: utf-8 -*-
from glob import glob
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
from kornia import Resize
from kornia.augmentation import CenterCrop
from kornia.color import Normalize
import kornia
from model.parsing.model import BiSeNet


class Outsketch_Folder_Set(data.Dataset):
    def __init__(self, opt: object, dirpath: object, use_par: object = True, b_checkpoint_path: object = "") -> object:
        '''
            need set opt.img_h and opt.img_w
        :param opt:
        :param dirpath:
        :param use_par:
        :param b_checkpoint_path:
        '''
        super(Outsketch_Folder_Set, self).__init__()
        self.root = opt.root
        self.imgResol = opt.input_size
        self.datalist = sorted(glob("{}/*.jpg".format(dirpath))+glob("{}/*.png".format(dirpath)), key=os.path.getctime)
        self.lens = len(self.datalist)
        self.opt = opt
        self.use_par = use_par
        if use_par:
            assert len(b_checkpoint_path) > 0
            self.B_Net_Model = BiSeNet(n_classes=opt.parsing_nc)
            self.__init_BNet__(b_checkpoint_path)

    def __getitem__(self, index):
        line = self.datalist[index]
        inputs = self.loadImg(line)
        return inputs

    def __len__(self):
        return self.lens

    def loadImg(self, line):
        src = Image.open(line).convert("RGB")
        if self.use_par:
            par = self.get_parsing(src)
            par = np.transpose(par, [1, 2, 0])
            par = kornia.image_to_tensor(par, keepdim=False).float()
            gauss = kornia.filters.GaussianBlur2d(kernel_size=(5, 5), sigma=(0.2, 0.2))
            par = gauss(par)[0]
        src = kornia.image_to_tensor(np.array(src, dtype=float)).float() / 255
        img = torch.cat([src, par])
        return img  # 合在一起操作，后再将其分开

    def __init_BNet__(self, ckpt_path):
        self.B_Net_Model.load_state_dict(torch.load(ckpt_path))
        self.B_Net_Model.eval()

    def get_parsing(self, img):
        w, h = img.size
        size_l = [h, w]
        size_l.sort()
        sml = size_l[0]
        lel = size_l[1]
        smlflg = False
        if sml == w:
            smlflg = True
        scal = 512 / lel
        to_size = round(scal * sml)
        if smlflg:
            d_tosize = (to_size, 512)
        else:
            d_tosize = (512, to_size)
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        to_tensor1 = transforms.Compose([
            Resize(d_tosize),
            CenterCrop((512, 512)),
            Normalize(torch.FloatTensor([0.485, 0.456, 0.406]), torch.FloatTensor((0.229, 0.224, 0.225)))
        ])
        to_tensor2 = transforms.Compose([
            CenterCrop(d_tosize),
            Resize((h, w))
        ])
        with torch.no_grad():
            img = to_tensor(img)
            img = torch.unsqueeze(img, 0)
            img = to_tensor1(img)
            parsing = self.B_Net_Model(img)[0]
            parsing = torch.softmax(parsing, dim=1)
            parsing = to_tensor2(parsing)
            parsing = torch.squeeze(parsing, 0).numpy()
        return parsing
