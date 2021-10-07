# encoding: utf-8
'''
因为SR的输出B''中有噪音的存在，所以R的提取效果不好，这里使用Init stage的A、B、B'和B''域数据对R进行微调，主要步骤：
1 得到A、B、B'和B''域数据
    这里的数据：A和B的数据在路径：'/home/ay3/houls/watermark_dataset/derain/IniStage/train or valid'
                B'的数据路径：
2 对R进行微调
'''

import argparse
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils.transformed as transforms
import torchvision.transforms as trans

from AS_parser import parameter_parser
from ImageFolderDataset import MyImageFolder
from models.HidingRes import HidingRes
import numpy as np
from PIL import Image
from vgg import Vgg16


def main():
    ############### define global parameters ###############
    global opt, optimizerR,  writer, logPath,  schedulerR, val_loader, smallestLoss,  mse_loss
    global trainpics, validpics, testpics, outlogs, outcodes, outckpts, runfolder

    opt = parameter_parser()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    # 为固定的卷积神经网络寻找最适合（快）计算的算法
    cudnn.benchmark = True

    ############  create the dirs to save the result #############
    cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    experiment_dir = cur_time
    result_root = "/home/ay3/houls/Deep-Model-Watermarking/result"
    stage = "AdverStage"
    result_dir = os.path.join(result_root, stage, experiment_dir)
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    pics_dir = os.path.join(result_dir, "pics")
    trainpics = os.path.join(pics_dir, 'train')
    validpics = os.path.join(pics_dir, "valid")
    testpics = os.path.join(pics_dir, "test")
    outlogs = os.path.join(result_dir, "outlogs")
    outcodes = os.path.join(result_dir, "outcodes")
    outckpts = os.path.join(result_dir, "outckpts")
    runfolder = os.path.join(result_dir, "runfolder")

    if not os.path.exists(outckpts):
        os.makedirs(outckpts)
    if not os.path.exists(trainpics):
        os.makedirs(trainpics)
    if not os.path.exists(validpics):
        os.makedirs(validpics)
    if not os.path.exists(testpics):
        os.makedirs(testpics)
    if not os.path.exists(outlogs):
        os.makedirs(outlogs)
    if not os.path.exists(outcodes):
        os.makedirs(outcodes)
    if not os.path.exists(runfolder):
        os.makedirs(runfolder)

    logPath = outlogs + '/%s_batchsz_%d_log.txt' % (stage, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(outcodes)
    writer = SummaryWriter(log_dir=runfolder)

    # DATA_DIR_root = '/data-x/g10/zhangjie/PAMI/datasets/debone_final/R_ft/'
    # DATA_DIR = os.path.join(DATA_DIR_root, opt.datasets)
    InitStage_dir = "/home/ay3/houls/watermark_dataset/derain/IniStage"
    SR_dir = "SRB2out"
    DATA_DIR = os.path.join(InitStage_dir, SR_dir,
                            f'{opt.num}imgs_concat/{opt.whichG}_netG')
    traindir = os.path.join(DATA_DIR, 'train')
    valdir = os.path.join(DATA_DIR, 'valid')
    secret_dir = "/home/ay3/houls/Deep-Model-Watermarking/secret"

    train_dataset = MyImageFolder(
        traindir,
        transforms.Compose([
            transforms.ToTensor(),
        ]))
    val_dataset = MyImageFolder(
        valdir,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                              shuffle=True, num_workers=int(opt.workers))

    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                            shuffle=True, num_workers=int(opt.workers))

    Rnet = HidingRes(in_c=3, out_c=3)
    Rnet.cuda()

    # setup optimizer
    optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
    schedulerR = ReduceLROnPlateau(
        optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

    #load the checkpoints
    # 载入在Init阶段训练好的模型
    #opt.Rnet ="/public/zhangjie/debone/HR/debone_flower/pth/R169.pth"
    # opt.Rnet ="result/derain_flower_Init/modelrun/outckpts/2021-09-19-18_51_28/netR_epoch_84_sumloss=1.073026_Rloss=0.052485.pth"
    opt.Rnet = "/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_Init/2021-09-30-11_20/modelrun/outckpts/netR191.pth"
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet)
    print_network(Rnet)

    # define loss
    mse_loss = nn.MSELoss().cuda()
    smallestLoss = 50000
    print_log(
        "adversarial training is beginning .......................................................", logPath)
    for epoch in range(opt.niter):
        train(train_loader,  epoch,  Rnet=Rnet)
        val_rloss, val_r_mseloss, val_r_consistloss, val_sumloss = validation(
            val_loader, epoch, Rnet=Rnet)
        schedulerR.step(val_rloss)

        # save the best model parameters
        if val_sumloss < globals()["smallestLoss"]:
            globals()["smallestLoss"] = val_sumloss
            # do checkPointing
            torch.save(Rnet.state_dict(),
                       '%s/netR%d.pth' % (
                           outckpts, epoch))

    writer.close()


def train(train_loader, epoch, Rnet):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Rlosses = AverageMeter()
    R_mselosses = AverageMeter()
    R_consistlosses = AverageMeter()
    SumLosses = AverageMeter()

    Rnet.train()

    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Rnet.zero_grad()
        this_batch_size = int(data.size()[0])
        # cover_img左边是不带有水印的图像(来自A和B)
        # 右边是带有水印的图像（来自B'和B''）
        cover_img = data[0:this_batch_size, :, :, :]
        cover_img_clean_A = cover_img[:, :, 0:256, 0: 256]
        cover_img_clean_B = cover_img[:, :, 0:256, 256: 256*2]
        cover_img_B1 = cover_img[:, :, 0:256, 256*2: 256*3]
        # cover_img_wm是B''域的图像
        cover_img_wm = cover_img[:, :, 0:256, 256*3: 256*4]

        # loader = transforms.Compose([ transforms.Resize(256,256), trans.Grayscale(num_output_channels=1),transforms.ToTensor() ])
        loader = transforms.Compose([transforms.ToTensor()])
        # /home/ay3/houls/Deep-Model-Watermarking/secret
        clean_img = Image.open(
            "/home/ay3/houls/Deep-Model-Watermarking/secret/clean.png")
        clean_img = loader(clean_img)
        clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)
        clean_img = clean_img[0:this_batch_size, :, :, :]
        secret_img = Image.open(
            "/home/ay3/houls/Deep-Model-Watermarking/secret/flower.png")
        secret_img = loader(secret_img)
        secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)
        secret_img = secret_img[0:this_batch_size, :, :, :]

        if opt.cuda:
            cover_img = cover_img.cuda()
            cover_img_wm = cover_img_wm.cuda()
            cover_img_clean_A = cover_img_clean_A.cuda()
            secret_img = secret_img.cuda()
            clean_img = clean_img.cuda()

        # 计算从B'和B''中提取出的水印和真实水印的loss
        extract_img_wm = Rnet(cover_img_wm)
        # betamse : 10000
        # wm_loss_sm
        # 计算从B2中提取出的水印和真实水印图像的loss
        errR_mse = opt.betamse * mse_loss(extract_img_wm, secret_img)
        # 从不带有水印的图像中提取内容
        extract_img_clean = Rnet(cover_img_clean_A)
        # 计算 clean wm_loss
        errR_clean = opt.betamse * mse_loss(extract_img_clean, clean_img)

        half_batchsize = int(this_batch_size / 2)
        # 计算一致损失（只计算了带有水印的图像样本）
        errR_consist = opt.betamse * \
            mse_loss(extract_img_wm[0:half_batchsize, :, :, :],
                     extract_img_wm[half_batchsize:half_batchsize*2, :, :, :])

        errR = errR_mse + opt.betacons * errR_consist + opt.betaclean * errR_clean

        err_sum = errR
        err_sum.backward()
        optimizerR.step()

        Rlosses.update(errR.data, this_batch_size)
        R_mselosses.update(errR_mse.data, this_batch_size)
        R_consistlosses.update(errR_consist.data, this_batch_size)

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        log = '[%d/%d][%d/%d]\t Loss_R: %.4f Loss_R_mse: %.4f Loss_R_consist: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
            Rlosses.val, R_mselosses.val, R_consistlosses.val,  SumLosses.val, data_time.val, batch_time.val)

        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)

        if i % 50 == 0:
            save_result_pic(this_batch_size, cover_img_wm.data[0], secret_img, extract_img_wm.data[0],
                            cover_img_clean_A.data[0], clean_img, extract_img_clean.data[0], epoch, i, trainpics)

    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + \
        " optimizerR_lr = %.8f " % (optimizerR.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_Rloss=%.6f\tepoch_R_mseloss=%.6f\tepoch_R_consistloss=%.6f\tepoch_sumLoss=%.6f" % (
        Rlosses.avg, R_mselosses.avg, R_consistlosses.avg, SumLosses.avg)

    print_log(epoch_log, logPath)

    writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/beta", opt.beta, epoch)

    writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
    writer.add_scalar('train/R_mse_loss', R_mselosses.avg, epoch)
    writer.add_scalar('train/R_consist_loss', R_consistlosses.avg, epoch)
    writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)


def validation(val_loader,  epoch, Rnet):
    print("#################################################### validation begin ########################################################")
    start_time = time.time()

    Rnet.eval()
    Rlosses = AverageMeter()
    R_mselosses = AverageMeter()
    R_consistlosses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    with torch.no_grad():

        for i, data in enumerate(val_loader, 0):
            data_time.update(time.time() - start_time)

            Rnet.zero_grad()

            this_batch_size = int(data.size()[0])
            cover_img = data[0:this_batch_size, :, :, :]
            cover_img_clean = cover_img[:, :, 0:256, 0:256]
            cover_img_wm = cover_img[:, :, 0:256, 256*3:256*4]

            # loader = transforms.Compose([ transforms.Resize(256,256), trans.Grayscale(num_output_channels=1),transforms.ToTensor() ])
            loader = transforms.Compose([transforms.ToTensor()])
            clean_img = Image.open(
                "/home/ay3/houls/Deep-Model-Watermarking/secret/clean.png")
            clean_img = loader(clean_img)
            clean_img = clean_img.repeat(this_batch_size, 1, 1, 1)
            clean_img = clean_img[0:this_batch_size, :, :, :]
            secret_img = Image.open(
                "/home/ay3/houls/Deep-Model-Watermarking/secret/flower.png")

            secret_img = loader(secret_img)
            secret_img = secret_img.repeat(this_batch_size, 1, 1, 1)
            secret_img = secret_img[0:this_batch_size, :, :, :]

            if opt.cuda:
                cover_img = cover_img.cuda()
                cover_img_wm = cover_img_wm.cuda()
                cover_img_clean = cover_img_clean.cuda()
                secret_img = secret_img.cuda()
                clean_img = clean_img.cuda()

            extract_img_wm = Rnet(cover_img_wm)
            errR_mse = opt.betamse * mse_loss(extract_img_wm, secret_img)

            extract_img_clean = Rnet(cover_img_clean)
            errR_clean = opt.betamse * mse_loss(extract_img_clean, clean_img)

            half_batchsize = int(this_batch_size / 2)
            errR_consist = opt.betamse * \
                mse_loss(extract_img_wm[0:half_batchsize, :, :, :],
                         extract_img_wm[half_batchsize:half_batchsize*2, :, :, :])

            errR = errR_mse + opt.betacons * errR_consist + opt.betaclean * errR_clean

            Rlosses.update(errR.data, this_batch_size)
            R_mselosses.update(errR_mse.data, this_batch_size)
            R_consistlosses.update(errR_consist.data, this_batch_size)

            if i % 50 == 0:
                save_result_pic(this_batch_size, cover_img_wm.data[0], secret_img, extract_img_wm.data[0],
                                cover_img_clean.data[0], clean_img, extract_img_clean.data[0], epoch, i, validpics)

    val_rloss = Rlosses.avg
    val_r_mseloss = R_mselosses.avg
    val_r_consistloss = R_consistlosses.avg
    val_sumloss = opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d]  val_Rloss = %.6f\t val_R_mseloss = %.6f\t val_R_consistloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch,  val_rloss, val_r_mseloss, val_r_consistloss, val_sumloss, val_time)

    print_log(val_log, logPath)

    writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
    writer.add_scalar('validation/R_mse_loss', R_mselosses.avg, epoch)
    writer.add_scalar('validation/R_consist_loss', R_consistlosses.avg, epoch)
    writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print(
        "#################################################### validation end ########################################################")
    return val_rloss, val_r_mseloss, val_r_consistloss, val_sumloss


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# 保存本次实验的代码
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)
    cur_work_dir, mainfile = os.path.split(main_file_path)

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


# print the training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic(this_batch_size, originalLabelvA, originalLabelvB, Container_allImg, secretLabelv, RevSecImg, RevCleanImgA, epoch, i, save_path):
    if not opt.debug:
        originalFramesA = originalLabelvA.resize_(
            this_batch_size, 3, opt.imageSize, opt.imageSize)
        originalFramesB = originalLabelvB.resize_(
            this_batch_size, 3, opt.imageSize, opt.imageSize)
        container_allFrames = Container_allImg.resize_(
            this_batch_size, 3, opt.imageSize, opt.imageSize)

        secretFrames = secretLabelv.resize_(
            this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames = RevSecImg.resize_(
            this_batch_size, 3, opt.imageSize, opt.imageSize)
        revCleanFramesA = RevCleanImgA.resize_(
            this_batch_size, 3, opt.imageSize, opt.imageSize)

        showResult = torch.cat([originalFramesA, originalFramesB, container_allFrames,
                                secretFrames, revSecFrames, revCleanFramesA], 0)

        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (
            save_path, epoch, i)

        vutils.save_image(showResult, resultImgName,
                          nrow=this_batch_size, padding=1, normalize=False)


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


if __name__ == '__main__':
    main()
