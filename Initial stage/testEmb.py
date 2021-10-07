'''

'''

from PIL import Image
import torch
import time
import os
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from models.HidingRes import HidingRes
from models.HidingUNet import UnetGenerator
import sys
sys.path.append("/home/ay3/houls/Deep-Model-Watermarking")
from SR.test_parser import parameter_parser
from SR.SRDataset import SRDataset

opt = parameter_parser()
root = "/home/ay3/houls/Deep-Model-Watermarking"
Hmodelpath = "/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20/modelrun/outckpts/netH191.pth"
Rmodelpath = '/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20/modelrun/outckpts/netR191.pth'
Rnet = HidingRes(in_c=3, out_c=3)
Rnet.load_state_dict(torch.load(Rmodelpath))
Hnet = UnetGenerator(input_nc=6, output_nc=3,
                     num_downs=7, output_function=nn.Sigmoid)
Hnet.load_state_dict(torch.load(Hmodelpath))
# test_dir = "/home/ay3/houls/watermark_dataset/test"
StageRoot = os.path.join(
    '/home/ay3/houls/watermark_dataset/derain', 'IniStage')
mode = "valid"
# StageOriDir: '/home/ay3/houls/watermark_dataset/derain/SrrStage/train'
StageOriDir = os.path.join(StageRoot, mode)
# StageOriDir = os.path.join("/home/ay3/houls/watermark_dataset/derain", mode)
test_loader = DataLoader(SRDataset(opt, StageOriDir),
                         batch_size=1, shuffle=False, num_workers=8)
# InitResultRoot ='/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20'
# 
testpicsdir = os.path.join(
            StageRoot, f'after_watermark/{opt.outnum}imgs_concat/netH191', mode)
# testpicsdir = "/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-10-05-21_22/testHemb"
# testpicsdir = os.path.join("/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20/pics", "testemb")
# testpicsdir = os.path.join('/home/ay3/houls/watermark_dataset/derain', 'AdverStage', mode)
if not os.path.exists(testpicsdir):
    os.makedirs(testpicsdir)

def test(test_loader, Hnet, Rnet, testpicsDir):
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    if torch.cuda.is_available():
        Hnet.cuda()
        Rnet.cuda()
    # Tensor type
    Tensor = torch.cuda.FloatTensor
    # if opt.cuda:
    #     Hnet = Hnet.cuda()
    #     Rnet = Rnet.cuda()
    # print_network(Hnet)
    with torch.no_grad():
        loader = transforms.Compose(
            [  # trans.Grayscale(num_output_channels=1),
                transforms.ToTensor(), ])
        clean_img = Image.open(os.path.join(root, "secret/clean.png"))
        #clean_img = Image.open("../secret/clean.png")
        clean_img = loader(clean_img)
        #secret_img = Image.open("../secret/flower.png")
        secret_img = Image.open(os.path.join(
            root, "secret/flower.png"))
        secret_img = loader(secret_img)
        for i, data in enumerate(test_loader, 0):
            Hnet.zero_grad()
            Rnet.zero_grad()
            cover_img_A = data['A']
            cover_img_B = data['B']
            this_batch_size = cover_img_A.size(0)

            secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)
            secret_img = secret_img[0:this_batch_size, :, :, :]
            clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)
            clean_img = clean_img[0:this_batch_size, :, :, :]

            if torch.cuda.is_available():
                cover_img_A = cover_img_A.cuda()
                cover_img_B = cover_img_B.cuda()
                secret_img = secret_img.cuda()
                clean_img = clean_img.cuda()
            # print(f'{cover_img_B.size(), secret_img.size()}')
            concat_img = torch.cat([cover_img_B, secret_img], dim=1)
            # container_img带有水印的灰度图像
            container_img = Hnet(concat_img)
            # container_img的3通道形式的图像
            #container_img_rgb = container_img.repeat(1, 1, 1, 1)
            # B的3通道形式的图像
            #cover_imgv_rgb = cover_imgv.repeat(1, 1, 1, 1)
            #cover_imgv_rgb.detach()
            # rev_secret_img 恢复出来的水印
            rev_secret_img = Rnet(container_img)
            # #rev_secret_img_rgb = rev_secret_img.repeat(1, 1, 1, 1)
            # # 从A和B中恢复出来的水印
            # clean_rev_secret_img_A = Rnet(cover_img_A)
            # clean_rev_secret_img_B = Rnet(cover_img_B)

            # if i % 1000 == 0:
            diff = 50 * (container_img - cover_img_B)
            with torch.no_grad():
                # image_tensor = torch.cat([cover_img_A, cover_img_B, container_img, diff, clean_rev_secret_img_A, clean_rev_secret_img_B, rev_secret_img, secret_img], axis=0)
                # save_result_pic(cover_img_A, cover_img_B, container_img, diff, clean_rev_secret_img_A, clean_rev_secret_img_B, rev_secret_img, secret_img, 'test', i , testpicsDir)
                save_result_pic(cover_img_A, cover_img_B,
                                container_img, 'test', i, testpicsDir)
    print("#################################################### test end ########################################################")


def save_result_pic(cover_img_A, cover_img_B, container_img,  epoch, i, save_path):
    this_batch_size = cover_img_A.size(0)
    originalFramesA = cover_img_A.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)
    originalFramesB = cover_img_B.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)
    container_allFrames = container_img.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)

    showResult = torch.cat(
        [originalFramesA, originalFramesB, container_allFrames], -1)
    resultImgName = '%s/ResultPics_epoch%s_batch%s.png' % (
        save_path, epoch, i)
    vutils.save_image(showResult, resultImgName,
                      nrow=this_batch_size, padding=0, normalize=False)
# save result pic and the coverImg filePath and the secretImg filePath
# def save_result_pic(cover_img_A, cover_img_B, container_img, diff, clean_rev_secret_img_A, clean_rev_secret_img_B, rev_secret_img, secret_img, epoch, i, save_path):

#     this_batch_size = cover_img_A.size(0)
#     originalFramesA = cover_img_A.reshape(
#         this_batch_size, 3, opt.imageSize, opt.imageSize)
#     originalFramesB = cover_img_B.reshape(
#         this_batch_size, 3, opt.imageSize, opt.imageSize)
#     container_allFrames = container_img.reshape(
#         this_batch_size, 3, opt.imageSize, opt.imageSize)

#     secretFrames = secret_img.reshape(
#         this_batch_size, 3, opt.imageSize, opt.imageSize)
#     revSecFrames = rev_secret_img.reshape(
#         this_batch_size, 3, opt.imageSize, opt.imageSize)
#     revCleanFramesA = clean_rev_secret_img_A.reshape(
#         this_batch_size, 3, opt.imageSize, opt.imageSize)
#     revCleanFramesB = clean_rev_secret_img_B.reshape(
#         this_batch_size, 3, opt.imageSize, opt.imageSize)
#     showResult = torch.cat([originalFramesA, originalFramesB, container_allFrames, diff, revCleanFramesA, revCleanFramesB, revSecFrames,
#                         secretFrames, ], -1)
#     resultImgName = '%s/ResultPics_epoch%s_batch%s.png' % (
#         save_path, epoch, i)
#     vutils.save_image(showResult, resultImgName,
#                         nrow=this_batch_size, padding=0, normalize=False)


test(test_loader, Hnet, Rnet, testpicsdir)
