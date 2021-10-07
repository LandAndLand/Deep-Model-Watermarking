import torch
import torch.nn as nn
# from torch.nn import init
import functools
import re
# from torch.autograd import Variable
import numpy as np
import os
from PIL import Image
import torchvision.utils as vutils


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def tensor2im(image_tensor, imtype=np.uint8):
    #print(f'image_tensor size util: {image_tensor.size()} ')
    image_numpy = image_tensor.detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0))

    return image_numpy.astype(imtype)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# 该函数作用是把log_info写入log_path指向的文件中


def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')


# def save_result_pic(Real_A, Real_B, Fake_B, index, epoch, save_path):
#     # print(f'Real_A shape: {Real_A.size()}')
#     images_tensor = torch.cat([Real_A, Real_B, Fake_B], axis=-1)
#     # print(f'images_tensor shape: {images_tensor.size()}')
#     image = tensor2im(images_tensor[4])
#     image_name = 'epoch%s_index%s_%s.png' % (
#         str(epoch), str(index), str(4))
#     image_path = os.path.join(save_path, image_name)
#     image_pil = Image.fromarray(image)
#     image_pil.save(image_path)
#     # for i in range(images_tensor.size(0)):
    #     image = tensor2im(images_tensor[i])
    #     image_name = 'epoch%s_index%s_%s.png' % (
    #         str(epoch), str(index), str(i))
    #     image_path = os.path.join(save_path, image_name)
    #     image_pil = Image.fromarray(image)
    #     image_pil.save(image_path)
def save_result_pic(images_tensor, img_name, epoch, save_path):
    # print(f'Real_A shape: {Real_A.size()}')
    # showResult = torch.cat([originalFramesA, originalFramesB, container_allFrames, ], -1)
    # print(f'save_result_pic: {img_name}')
    if epoch!= "":
        resultImgName =os.path.join(save_path, f"{epoch}_{img_name}")
    else:
        resultImgName =os.path.join(save_path, img_name)
    vutils.save_image(images_tensor, resultImgName,
                        nrow=1, padding=0, normalize=False)

def save_result_pic_single(images_tensor, img_name, epoch, save_path):
    # print(f'Real_A shape: {Real_A.size()}')
    for i in range(images_tensor.size(0)):
        name = img_name[i].split(".")[0]
        #print(f'img name: {name}')
        image = tensor2im(images_tensor[i])
        image_name = '%sepoch_%s.png' % (
            str(epoch), name)
        image_path = os.path.join(save_path, image_name)
        image_pil = Image.fromarray(image)
        image_pil.save(image_path)
def sort_list(list_model):
    list_model.sort(key=lambda x: int(
        re.match('^[a-z]*[A-Z]([0-9]+)', x).group(1)))

def GetBestModel(model_dir, mode='H'):
    if mode == 'H':
        Hmodels = []
        for model in os.listdir(model_dir):
            if "netH" in model:
                Hmodels.append(model)
        sort_list(Hmodels)
        print(Hmodels[-1])
        return Hmodels[-1]
    elif mode == 'R':
        Rmodels = []
        for model in os.listdir(model_dir):
            if "netR" in model:
                Rmodels.append(model)
        sort_list(Rmodels)
        print(Rmodels[-1])
        return Rmodels[-1]
    else:
        Dmodels = []
        for model in os.listdir(model_dir):
            if "netD" in model:
                Dmodels.append(model)
        sort_list(Dmodels)
        print(Dmodels[-1])
        return Dmodels[-1]
