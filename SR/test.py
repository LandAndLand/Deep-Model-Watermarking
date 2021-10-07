from posixpath import realpath
import time
import torch
import os
#from util.visualizer import Visualizer
from pdb import set_trace as st
#from ssim import SSIM
from test_parser import parameter_parser
# from SR_parser import parameter_parser
from SRDataset import SRDataset
from tqdm import tqdm
from models import networks
from utils import AverageMeter, tensor2im, print_log, save_result_pic, GetBestModel
from models.losses import init_loss
'''
使用训练好的SR模型来对Init Stage的图像进行处理， 也就是从Init Stage的A得到B''
1 输入Init Stage的A域图像作为输入： "/home/ay3/houls/watermark_dataset/derain/IniStage" + 'train/valid';
2 载入训练好的SR模型：如/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-09-27-22_30/checkpoints/2_netG.pth
3 得到InitStage数据集下的fake_B, 分别将A、B、B1、B2拼接起来存到一个图像中 
	存放到: "/home/ay3/houls/watermark_dataset/derain/IniStage/after_watermark/4imgs_concat/netH58/train" + 'train/valid';

这里在为Adver Stage准备数据时，输入的是"/home/ay3/houls/watermark_dataset/derain/IniStage/after_watermark/3imgs_concat/{modelname}/train or valid的数据" 
输出到"/home/ay3/houls/watermark_dataset/derain/IniStage/SRB2out/4imgs_concat/{modelname}'
python test.py --outnum 4

在测试训练号的SR model时， 输入的是"/home/ay3/houls/watermark_dataset/derain/test
输出到/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-10-04-14_37/test_R
python tets.py --num 2 --outnum 3
'''

def test(dataset_dir, netG):
	test_dataset = SRDataset(opt, dataset_dir)
	test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=int(opt.nThreads))
	test_log_path = os.path.join(result_dir, f'testlog_{modelname}.txt')
	print_log(f"dataset_dir for input : {dataset_dir}", test_log_path)
	
	t = torch.cuda.FloatTensor if len(opt.gpu_ids) > 0 else torch.Tensor
	#discLoss, contentLoss = init_loss(opt, t)
	start_time = time.time()
	netG.eval()
	for i, data in tqdm(enumerate(test_loader)):
		input_A = data['A']
		real_B = data['B']
		#print(f'input a size: {input_A.size()}')
		this_batch_size = int(input_A.size()[0])
		img_path = data['A_paths'][1]
		if opt.num == 3:
			# 带有水印的图像 B1
			input_B1 = data['B1']
			if len(opt.gpu_ids) > 0:
				input_A = input_A.cuda()
				input_B1 = input_B1.cuda()
				real_B = real_B.cuda()
		elif opt.num == 2:	
			#input_B1 = data['B1']
			if len(opt.gpu_ids) > 0:
				input_A = input_A.cuda()
				real_B = real_B.cuda()
		else:
			print('you need to chose num=2 or 3')
		# fake_B是使用sr model得到的B2图像
		fake_B = netG.forward(input_A)
		if opt.outnum == 4:
			# print(f'img size: {input_A.size(), real_B.size(), input_B1.size(), fake_B.size()}')
			images_tensor = torch.cat([input_A, real_B, input_B1, fake_B], axis=-1)
			save_result_pic(images_tensor, img_path[0], "inittest", SRoutB2_dir)
		elif opt.outnum ==3:
			# print(f'img size: {input_A.size, input_B1.size(), fake_B.size()}')
			images_tensor = torch.cat([input_A, real_B, fake_B], axis=-1)
			save_result_pic(images_tensor, img_path[0], "inittest", SRoutB2_dir)
		else:
			print(f'opt outnum: {opt.outnum}')
		# save_result_pic_test(fake_B, i, 1, test_result_dir_single)
	# batch_time.update(time.time() - bt)
	data_time.update(time.time() - start_time)
	# 计算VGG_14(real_B)和VGG_14(fake_B)之间的MSE距离
	# loss_G_Content_test = contentLoss.get_loss(
    #         fake_B, input_B) * opt.lambda_A
	#loss_G_Content_testlog.update(loss_G_Content_test.data, this_batch_size)
	start_time = time.time()
	log = '[%d]\t datatime: %.4f \t' % (
		len(test_loader), data_time.val)
	print_log(log, test_log_path)
	# save_result_pic_test(input_A, input_B,
    #                   fake_B, i, 1, test_result_dir)

if __name__ == "__main__":
	global opt, outroot, modelname, SRoutB2_dir, modelpath
	data_time = AverageMeter()
	opt = parameter_parser()
	# result/derain_flower_SR
	outroot = opt.outroot
	# "/home/ay3/houls/watermark_dataset/derain"
	dataset_root = opt.datasetdir

	# /home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-0**
	result_dir = os.path.join(outroot, opt.datetime)
	checkpoints_dir = os.path.join(result_dir, 'checkpoints')
	# 219_netG.pth
	modelname = '%s_netG.pth'%(opt.which_epoch)
	modelpath = os.path.join(checkpoints_dir, modelname)
	netG = networks.define_G(
            3, 3, 64, 'resnet_9blocks', 'instance',
            not opt.no_dropout, False
        )
	netG.load_state_dict(torch.load(modelpath))
	if len(opt.gpu_ids) > 0:
		netG.cuda()
	# for mode in ['train', 'valid']:
	# dataset_dir = os.path.join(result_dir, 'test2')
	# initroot: "/home/ay3/houls/watermark_dataset/derain/IniStage"
	mode="valid"
	# dataset_dir = os.path.join(opt.initroot, f'after_watermark/{opt.num}imgs_concat/netH{opt.whichH}', mode)
	# dataset_dir = os.path.join("/home/ay3/houls/watermark_dataset/derain", 'test')	
	dataset_dir = os.path.join("/home/ay3/houls/watermark_dataset/derain/SrrStage", 'valid')	
	# dataset_dir = os.path.join(result_dir, 'test')	
	# dataset_dir = os.path.join(result_dir, 'testHemb')				
	
	# print(f'dataset dir : {dataset_dir}')
	SRoutB2_dir = os.path.join(opt.initroot, f'SRB2outFalseValid/4imgs_concat/{modelname.split(".")[0]}', mode)
	# SRoutB2_dir = os.path.join(result_dir, 'test')
	# SRoutB2_dir = os.path.join(result_dir, 'testHemb_R')

	if not os.path.exists(SRoutB2_dir):
		os.makedirs(SRoutB2_dir)
	test(dataset_dir, netG)
