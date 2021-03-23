from kornia.augmentation import CenterCrop
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import kornia
from myutils.summary_util import *
from config.SAND_pix_opt import TestOptions
from data.out_dir_dataset import Outsketch_Folder_Set
from model.total_model.SAND_base_lighting_model import SAND_pix_Gen_Parsing
import os
opt = TestOptions().parse()

def apply_tranform(tensors):
    y_trans_arr = []
    # share transform
    y_trans_arr.append(kornia.geometry.Resize((opt.img_h, opt.img_w)))
    y_trans_arr.append(kornia.augmentation.CenterCrop((opt.input_size, opt.input_size)))
    y_trans_method = torch.nn.Sequential(*y_trans_arr)
    y_trans_method = y_trans_method
    # split
    org_nc = opt.image_nc
    org_nc += opt.parsing_nc
    tensors = y_trans_method(tensors)
    # normalized
    src_img = tensors[:, :opt.image_nc]
    src_par = tensors[:, opt.image_nc:org_nc]
    src_img = kornia.color.Normalize(0.5, 0.5)(src_img)
    src = torch.cat([src_img, src_par], dim=1)
    return src


def dataloader_test(dataloader, model_our, tar_path, device="cuda"):
    def ddde(img):
        if img.size(1) == 1:
            img = torch.cat([img, img, img], dim=1)
        # trans
        trans = CenterCrop((200,200))
        img = trans(img)
        img = tensor_to_image(img.cpu())
        img = ToPILImage()(img.squeeze(0))
        return img

    for i, batch in enumerate(dataloader):
        print(i)
        x = batch
        x = apply_tranform(x)
        x = x.to(device)
        par = x[:, 3:]
        x = x[:, :3]
        gen_img_our = model_our.Generator(x, par, par)
        gen_img_our = ddde(gen_img_our[0])
        gen_img_our.save("{}/{}.jpg".format(tar_path, i + 1))


def evluate_outdata():
    def load_model_check(check_point_path, model, device):
        checkpoint = torch.load(check_point_path, map_location=lambda storage, loc: storage)['state_dict']
        model_dict = model.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() if (k in model_dict) and ("G" in k)}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        model.eval()
        model.to(device)
        return model

    os.makedirs(opt.results_dir, exist_ok=True)
    model_our = SAND_pix_Gen_Parsing(opt)
    device = "cuda:2"
    model_our = load_model_check(opt.checkpoint_dir, model_our, device)
    dataset = Outsketch_Folder_Set(opt, opt.data_dir, b_checkpoint_path=opt.bisenet_dir)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8, shuffle=False)
    dataloader_test(dataloader, model_our, opt.results_dir, device)


if __name__ == '__main__':
    evluate_outdata()
