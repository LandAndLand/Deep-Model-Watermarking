'''
用于测试在InitStage阶段训练好的R是否可以提取出SR阶段model输出的B''图像中的水印
1 读取result/derain_flower_SR/2021-09-27-22_30/test/Init_train或者Init_valid的图像作为待测试数据
    /home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-09-27-22_30/test/Init_train/114/epoch1_index0_0.png
2 构建R模型并读取Init Stage阶段训练好的R模型 路径： result/derain_flower_Init/modelrun/outckpts/2021-09-24-21_50_35/netR_epoch_193_sumloss=1.914812_Rloss=0.015234.pth
    R绝对路径： /home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/modelrun/outckpts/2021-09-24-21_50_35/netR_epoch_193_sumloss=1.914812_Rloss=0.015234.pth
3 使用R来处理数据，提取出水印图像，并保存到 result/derain_flower_SR/2021-09-27-22_30/test_R中
    绝对路径： /home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-09-27-22_30/test
注意：在该阶段，SR模型能够接触到A、B'、B''数据，可以
'''
import torchvision.utils as vutils
import argparse
import os
from numpy.core.fromnumeric import repeat
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from models.HidingRes import HidingRes
import sys
sys.path.append("/home/ay3/houls/Deep-Model-Watermarking")
from SR.SRDataset import SRDataset
from SR.utils import save_result_pic as save_result_pic1
def save_result_pic(opt, cover_img_A, cover_img_B, container_img, clean_rev_secret_img_A, clean_rev_secret_img_B, rev_secret_img, secret_img, imgname, save_path):
    this_batch_size = cover_img_A.size(0)
    originalFramesA = cover_img_A.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)
    originalFramesB = cover_img_B.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)
    container_allFrames = container_img.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)

    secretFrames = secret_img.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)
    revSecFrames = rev_secret_img.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)
    revCleanFramesA = clean_rev_secret_img_A.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)
    revCleanFramesB = clean_rev_secret_img_B.reshape(
        this_batch_size, 3, opt.imageSize, opt.imageSize)
    showResult = torch.cat([originalFramesA, originalFramesB, container_allFrames, revCleanFramesA, revCleanFramesB, revSecFrames,
                        secretFrames, ], 0)
    resultImgName = f'%s/{imgname}' % (save_path)
    vutils.save_image(showResult, resultImgName,
                        nrow=this_batch_size, padding=1, normalize=False)

def test_R(mode='train'):
    parser =  argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/home/ay3/houls/watermark_dataset/derain')
    parser.add_argument('--outroot', default='/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-10-05-21_22')
    #parser.add_argument('--inputdir', default='result/derain_flower_SR/2021-09-27-22_30/test/Init_train', help='the path of input( from SR output)')
    # parser.add_argument('--Rmodelpath', default='/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20/modelrun/outckpts/netR191.pth', help="the R model's path")
    parser.add_argument('--Rmodelpath', default="/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20/modelrun/outckpts/netR191.pth")
    parser.add_argument('--cuda', default=True, help='if use gpu or not')
    parser.add_argument('--imageSize', default=256)
    # parser.add_argument('--mode', default='train', help="the folder of SR output to test on R")
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--batchsize', type = int, default=4)
    parser.add_argument('--adverStage', default="/home/ay3/houls/watermark_dataset/derain/AdvStage")
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument('--outnum', type=int , default=4)
    parser.add_argument('--whichG', default="219")
    args, unknow = parser.parse_known_args()
    # 输入数据路径为： 'result/derain_flower_SR/2021-09-27-22_30/test/Init_train'
    # input_dir = '/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20/pics/testemb' 
    # input_dir = os.path.join("/home/ay3/houls/watermark_dataset/derain/IniStage/after_watermark", f'3imgs_concat/netH191/train')
    # input_dir = os.path.join(args.outroot, 'test')
    input_dir = os.path.join(args.dataset, f"IniStage/SRB2outTrueValid/3imgs_concat/{args.whichG}_netG", "valid")
    # input_dir = os.path.join(args.dataset, f'IniStage/after_watermark/{args.num}imgs_concat/netH191', "train")
    # input_dir = os.path.join("/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20/pics/after_watermark")
    # input_dir = os.path.join("/home/ay3/houls/watermark_dataset/derain/SrrStage/after_watermark/3imgs_concat/netH191/valid")    
    # input_dir = '/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-10-05-21_22/testHemb_R'
    # test_R1:是只识别B1的结果
    # test_R2是只识别B2的结果   
    test_R_dir = os.path.join(args.outroot, 'SRB2outTrueValid')
    if not os.path.exists(test_R_dir):
        os.makedirs(test_R_dir)
    Rdataset = SRDataset(args, input_dir)
    # Rdataset = threeDataset(args, input_dir)
    Rdataloader = torch.utils.data.DataLoader(Rdataset, batch_size=1, shuffle=False, num_workers=8)
    Rnet = HidingRes(in_c=3, out_c=3)
    Rnet.load_state_dict(torch.load(args.Rmodelpath))
        
    loader = transforms.Compose([transforms.ToTensor(),])
                            #   transforms.Normalize((0.5,0.5,0.5 ),
                            #                   (0.5,0.5,0.5 ))])
    secret_path = "/home/ay3/houls/Deep-Model-Watermarking/secret/flower.png"
    secret_img = Image.open(secret_path)
    secret_img = loader(secret_img)
    secret_img = secret_img.repeat(1, 1, 1, 1)
    if args.cuda:
        Rnet.cuda()
        secret_img = secret_img.cuda()
    Rnet.eval()
    for i, data in tqdm(enumerate(Rdataloader)):
        clean_A = data['A']
        clean_B = data['B']
        image_name = data['A_paths'][1][0]
        if args.num == 3:
            SRoutput = data['B1']
            if args.cuda:
                SRoutput = SRoutput.cuda()
                clean_A = clean_A.cuda()
                clean_B = clean_B.cuda()
            R_clean_A = Rnet(clean_A)
            R_clean_B = Rnet(clean_B)
            R_SR = Rnet(SRoutput)
            # images_tensor = torch.cat([clean_A, clean_B,  SRoutput, R_SR], axis=-1)
            images_tensor = torch.cat([clean_A, R_clean_A, clean_B, R_clean_B, SRoutput, R_SR, secret_img], axis=-1)
            save_result_pic1(images_tensor, image_name, "test_R_3", test_R_dir)
            # save_result_pic(args, clean_A, clean_B, SRoutput, R_clean_A, R_clean_B, R_SR, secret_img, image_name, test_R_dir)
        elif args.num==4:
            Hemb = data['B1']
            SRoutput = data['B2']
            if args.cuda:
                Hemb = Hemb.cuda()
                SRoutput = SRoutput.cuda()
                clean_A = clean_A.cuda()
                clean_B = clean_B.cuda()
            # R_clean_A = Rnet(clean_A)
            # R_clean_B = Rnet(clean_B)
            R_B1 = Rnet(Hemb)
            R_SR = Rnet(SRoutput)
            images_tensor = torch.cat([clean_A, clean_B, Hemb, SRoutput, R_B1, R_SR, secret_img], axis=-1)
            save_result_pic1(images_tensor, image_name, "test_R_4", test_R_dir)
        else:
            R_B1 = Rnet(clean_B)
            images_tensor = torch.cat([clean_A, clean_B, R_B1, secret_img], axis=-1)
            save_result_pic1(images_tensor, image_name, "test_R_2", test_R_dir)
        # 按照1：clean_B(不带有水印的来自B域的ground-truth图像)，
        # 2： 从clean_B中提取的水印R_clean_B
        # 3： SRoutput（SR模型的输出 ）
        # 4： R_SR（从SRoutput中提取出的水印）
        # 5 真实水印图像
        # 来排列
        # print(f'size: {clean_A.size(), clean_B.size(), R_clean_A.size(), R_clean_B.size(), SRoutput.size(), R_SR.size(), secret_img.size()}')
        # images_tensor = torch.cat([clean_A, R_clean_A, clean_B, R_clean_B, SRoutput, R_SR, secret_img], axis=-1)
        # save_result_pic(images_tensor, image_name, "test_R", test_R_dir)
        # # if i % 10 ==0:
        #     break
if __name__ == "__main__":
    test_R()





