'''
主要思路:
使用Init stage数据集训练H和R后，用训练好的H对Srr stage数据集进行嵌入水印处理（对该数据集中的B域图像嵌入水印）;
然后使用Srr 数据集 的after-watermark数据（A和B'域的数据）对SR模型进行训练；
最后，使用训练好的SR模型对Init stage的数据集进行处理，主要就是将Init stage的A域数据输入到训练好的SR模型中，通过SR模型达到derain效果，
    同时又希望SR模型的输出中能够被训练好的R提取出水印
主要步骤：
    1 使用训练好的H对Srr stage数据集进行水印嵌入处理,得到B'数据；
    2 使用Srr stage的after-watermark数据（A和B'域的数据）对SR模型进行训练（train.py主要工作）；
    3 使用训练好的SR模型对Init stage数据集的A域数据进行处理，得到B''数据（test.py主要工作）
    4 使用在Init stage阶段训练好的R来提取B''数据中的水印（test_R.py主要工作），但是发现提取出来的水印效果并不理想。
    所以，衍生出了Adversarial stage阶段，使用A、B、B'和B''数据对R进行微调，增强R的提取能力
注意：输入的数据集是ABB1格式的3img_concat图像，
        路径： '/home/ay3/houls/watermark_dataset/derain/SrrStage/after_watermark/3imgs_concat/netH58/train ot valid'
        其中可以修改的是netH*
    输出4img_concat图像 ，这里直接输出到result对应的文件即可
'''
import time
import os
import torch
from SRDataset import SRDataset
from multiprocessing import freeze_support
from SR_parser import parameter_parser
from tqdm import tqdm
from models import networks
from models.losses import init_loss
from utils import weights_init, AverageMeter, tensor2im, print_log, save_result_pic
from util.metrics import PSNR, SSIM
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image


def main():
    global opt, outroot, result_dir, writer, logPath, discLoss, contentLoss, optimizer_D, optimizer_G, train_result_dir, valid_result_dir
    opt = parameter_parser()
    t = torch.cuda.FloatTensor if len(opt.gpu_ids) > 0 else torch.Tensor
    discLoss, contentLoss = init_loss(opt, t)
    # result/derain_flower_SR'
    outroot = opt.outroot
    cut_time = time.strftime('%Y-%m-%d-%H_%M', time.localtime())
    result_dir = os.path.join(outroot, cut_time)
    checkpoints_dir = os.path.join(result_dir, 'checkpoints')
    train_result_dir = os.path.join(result_dir, 'train')
    valid_result_dir = os.path.join(result_dir, 'valid')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(train_result_dir):
        os.makedirs(train_result_dir)
    if not os.path.exists(valid_result_dir):
        os.makedirs(valid_result_dir)
    writer = SummaryWriter(log_dir=result_dir, comment='**')
    logPath = result_dir + '_batchsz_%d_log.txt' % (opt.batchSize)
    print_log(str(opt), logPath)
    # "/home/ay3/houls/watermark_dataset/derain/SrrStage/after_watermark..."
    after_watermark_for_input_dir = opt.dataroot
    train_dir = os.path.join(after_watermark_for_input_dir,  f'{opt.num}imgs_concat/netH{opt.which_epoch}/train')
    valid_dir= os.path.join(after_watermark_for_input_dir,  f'{opt.num}imgs_concat/netH{opt.which_epoch}/valid')
    train_dataset = SRDataset(opt, train_dir)
    print(f"train dataset: {train_dir}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.nThreads)
    )
    valid_dataset = SRDataset(opt, valid_dir)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(opt.nThreads)
    )
# 初始化了网络结构和optimizer，并且被放到了GPU上（在network.py里面放的）
    use_parallel = False
    use_sigmoid = opt.gan_type == 'gan'
    netG = networks.define_G(
        opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
        True, True
    )
    netD = networks.define_D(
        opt.output_nc, opt.ndf, opt.which_model_netD,
        opt.n_layers_D, opt.norm, use_sigmoid)
    # initialize optimizers
    optimizer_G = torch.optim.Adam(
        netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D = torch.optim.Adam(
        netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    schedulerG = ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.2, patience=5, verbose=True)
    schedulerD = ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.2, patience=5, verbose=True)
    if len(opt.gpu_ids) > 0:
        netG.cuda()
        netD.cuda()
    netG.apply(weights_init)
    netD.apply(weights_init)
    smallestLoss = 3000
    for epoch in tqdm(range(opt.niter), desc=f" {opt.niter} epochs"):
        train(opt, epoch, train_loader, netG, netD)
        lossG, lossGAN, lossContent, lossD = validation(
            opt, epoch, valid_loader, netG, netD)
        schedulerG.step(lossG)
        schedulerD.step(lossD)
        if lossG < smallestLoss:
            smallestLoss = lossG
            torch.save(netG.state_dict(),
                       '%s/%d_netG.pth' % (
                checkpoints_dir, epoch+1))
            torch.save(netD.state_dict(),
                       '%s/%d_netD.pth' % (
                checkpoints_dir, epoch+1))

def train(opt, epoch, train_loader, netG, netD):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_G_log = AverageMeter()
    loss_G_GAN_log = AverageMeter()
    loss_G_Content_log = AverageMeter()
    loss_D_log = AverageMeter()

    netG.train()
    netD.train()
    # range(1, 150+150)
    start_time = time.time()
    for i, data in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - start_time)
        input_A = data['A']
        this_batch_size = int(input_A.size()[0])
        # num==3是按照ABB1存储的
        if opt.num == 3:
            input_B1 = data['B1']
            # 训练SR用不到real_B
            real_B = data['B']
            if len(opt.gpu_ids) > 0:
                real_B = real_B.cuda()
        # num==2是按照AB'存储的（这里的data['B']其实是B1，带有水印）
        elif opt.num == 2:
            input_B1 = data['B']
        image_paths = data['A_paths']
        # print(f'train image path: {image_paths[1]}')
        # print(f'input A shape: {input_A.size()}')
        # print(f'input B shape: {input_B.size()}')
        # print(f'image path: {image_paths}')
        if len(opt.gpu_ids) > 0:
            input_A = input_A.cuda()
            input_B1 = input_B1.cuda()
            # image_paths = image_paths.cuda()
        fake_B = netG.forward(input_A)

        optimizer_G.zero_grad()
        # 计算D对fake_B的预测结果与1之间的L1距离
        loss_G_GAN = discLoss.get_g_loss(netD, input_A, fake_B)
        # Second, G(A) = B
        # 计算VGG_14(real_B)和VGG_14(fake_B)之间的MSE距离
        loss_G_Content = contentLoss.get_loss(fake_B, input_B1) * opt.lambda_A
        loss_G = loss_G_GAN + loss_G_Content
        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        # loss_D是计算D对fake_B的预测结果和0的L1距离， 以及 D对real_B的预测结果和1的L1距离
        # D的损失越小，证明其判断真假的能力越强
        loss_D = discLoss.get_loss(netD, input_A, fake_B, input_B1)
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        loss_G_log.update(loss_G.data, this_batch_size)
        loss_D_log.update(loss_D.data, this_batch_size)
        loss_G_GAN_log.update(loss_G_GAN.data, this_batch_size)
        loss_G_Content_log.update(loss_G_Content.data, this_batch_size)

        batch_time.update(time.time() - start_time)
        start_time = time.time()
        log = '[%d/%d][%d/%d]\tloss_G_GAN: %.4f loss_G_Content: %.4f loss_G: %.4f loss_D: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
            loss_G_GAN_log.val, loss_G_Content_log.val, loss_G_log.val, loss_D_log.val, data_time.val, batch_time.val)
        # 每50个样本在控制台打印一次model运行信息
        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            # 只把log写入log 文件，不在控制台打印信息
            print_log(log, logPath, console=False)
        # 100
        if i % opt.resultPicFrequency == 0:
        # if i % 1 == 0:
            with torch.no_grad():
                if opt.outnum == 4:
                    # 按照A、B、B1、B2存储
                    images_tensor = torch.cat([input_A, real_B, input_B1, fake_B], axis=-1)
                    save_result_pic(
                        images_tensor, (image_paths[1])[0], epoch, train_result_dir)
                elif opt.outnum ==3 :
                    # 按照A、B1、B2存储
                    images_tensor = torch.cat([input_A, input_B1, fake_B], axis=-1)
                    save_result_pic(
                        images_tensor, (image_paths[1])[0], epoch, train_result_dir)
                
        # 100
        if i % opt.display_freq == 0:
            # 获得real_A, Fake_B, Real_B的图像
            fB = tensor2im(fake_B[0].data)
            rB = tensor2im(input_B1[0].data)
            psnrMetric = PSNR(fB, rB)
            # print('PSNR on Train = %f' % psnrMetric)
            print_log('PSNR on Train = %f' % psnrMetric, logPath)
    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerG_lr = %.8f  optimizerD_lr = %.8f" % (
        optimizer_G.param_groups[0]['lr'], optimizer_D.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_loss_G_GAN=%.6f\t epoch_loss_G_Content=%.6f\t epoch_loss_G=%.6f\t epoch_loss_D=%.6f" % (
        loss_G_GAN_log.avg, loss_G_Content_log.avg, loss_G_log.avg, loss_D_log.avg)
    print_log(epoch_log, logPath)

    writer.add_scalar('train/G_loss', loss_G_log.avg, epoch)
    writer.add_scalar('train/G_GAN_loss', loss_G_GAN_log.avg, epoch)
    writer.add_scalar('train/G_Content_loss', loss_G_Content_log.avg, epoch)
    writer.add_scalar('train/D_loss', loss_D_log.avg, epoch)


def validation(opt, epoch, valid_loader, netG, netD):
    print(
        "#################################################### validation begin ########################################################")
    if not os.path.exists(valid_result_dir):
        os.makedirs(valid_result_dir)
    start_time = time.time()
    val_time = AverageMeter()
    data_time = AverageMeter()
    loss_G_vallog = AverageMeter()
    loss_G_GAN_vallog = AverageMeter()
    loss_G_Content_vallog = AverageMeter()
    loss_D_vallog = AverageMeter()

    netG.eval()
    netD.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(valid_loader)):
            data_time.update(time.time() - start_time)
            if opt.num == 3:
                input_A = data['A']
                real_B = data['B']
                input_B1 = data['B1']
                if len(opt.gpu_ids) > 0:
                    real_B = real_B.cuda()
            # num==2是按照AB'存储的（这里的data['B']其实是B1，带有水印）
            elif opt.num == 2:
                input_B1 = data['B']
                input_A = data['A']
            this_batch_size = int(input_A.size()[0])
            image_paths = data['A_paths']
            # print(f'valid image path: {image_paths[1]}')
            if len(opt.gpu_ids) > 0:
                input_A = input_A.cuda()
                input_B1 = input_B1.cuda()
                # image_paths = image_paths.cuda()
            fake_B = netG.forward(input_A)
            loss_G_GAN = discLoss.get_g_loss(netD, input_A, fake_B)
            # Second, G(A) = B
            # 计算VGG_14(real_B)和VGG_14(fake_B)之间的MSE距离
            loss_G_Content = contentLoss.get_loss(
                fake_B, input_B1) * opt.lambda_A
            loss_G = loss_G_GAN + loss_G_Content
            loss_D = discLoss.get_loss(netD, input_A, fake_B, input_B1)

            loss_G_vallog.update(loss_G.data, this_batch_size)
            loss_D_vallog.update(loss_D.data, this_batch_size)
            loss_G_GAN_vallog.update(loss_G_GAN.data, this_batch_size)
            loss_G_Content_vallog.update(loss_G_Content.data, this_batch_size)
            # opt.display_freq ： 100
            if i % opt.display_freq == 0:
                with torch.no_grad():
                    if opt.outnum == 4:
                        # 按照A、B、B1、B2存储
                        valid_images_tensor = torch.cat([input_A, real_B, input_B1, fake_B], axis=-1)
                        save_result_pic(
                            valid_images_tensor, (image_paths[1])[0], epoch, valid_result_dir)
                    elif opt.outnum ==3 :
                        # 按照A、B1、B2存储
                        valid_images_tensor = torch.cat([input_A, input_B1, fake_B], axis=-1)
                        save_result_pic(
                            valid_images_tensor, (image_paths[1])[0], epoch, valid_result_dir)
            if i % opt.display_freq == 0:
                # 获得real_A, Fake_B, Real_B的图像
                fB = tensor2im(fake_B[0].data)
                rB = tensor2im(input_B1[0].data)
                psnrMetric = PSNR(fB, rB)
                # print('PSNR on Train = %f' % psnrMetric)
                print_log('PSNR on valid = %f' % psnrMetric, logPath)
    val_time = time.time() - start_time
    val_log = "validation[%d] val_time = %d \t val_loss_G_GAN = %.6f\t val_loss_G_Content = %.6f\t val_loss_G = %.6f\t val_loss_D = %.6f\t " % (
        epoch, val_time, loss_G_GAN_vallog.avg, loss_G_Content_vallog.avg, loss_G_vallog.avg, loss_D_vallog.avg)
    print_log(val_log, logPath)
    print("#################################################### validation end ########################################################")
    return loss_G_vallog.val, loss_G_GAN_vallog.val, loss_G_Content_vallog.val, loss_D_vallog.val


if __name__ == '__main__':
    main()
