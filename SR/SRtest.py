import os
import torch
from tqdm.std import tqdm

from models import networks
# from SRDataset import SRDataset
from SRTwo import SRTwo
from SR_parser import parameter_parser
from tqdm import tqdm
from utils import save_result_pic

opt = parameter_parser()
stage = "IniStage"
dataset_dir = "/home/ay3/houls/watermark_dataset"
StageDir = os.path.join(dataset_dir, stage)
mode = "train"
# "/home/ay3/houls/watermark_dataset/derain/IniStage/train"
modeStageDir = os.path.join(StageDir, mode)
# testDir = os.path.join(dataset_dir, "test")

result_root = "/home/ay3/houls/Deep-Model-Watermarking/result"
result_stage = "derain_flower_SR"
result_time = "2021-10-05-21_22"
result_dir = os.path.join(result_root, result_stage, result_time)
modelname = "219_netG.pth"
modelpath = os.path.join(result_dir, 'checkpoints', modelname)
input_name = "test"
# 输入到SR model中的路径
# input_dir = os.path.join(result_dir, input_name)
input_dir = os.path.join(dataset_dir, "test")
output_dir = os.path.join(result_dir, "SRout_one")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
netG = networks.define_G(
    3, 3, 64, 'resnet_9blocks', 'instance',
    not opt.no_dropout, True
)
netG.load_state_dict(torch.load(modelpath))
netG.cuda()
netG.eval()
dataset = SRTwo(input_dir)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=int(opt.nThreads))
for i, data in tqdm(enumerate(data_loader)):
    input_A = data['A'].cuda()
    real_B = data['B'].cuda()
    this_batch_size = int(input_A.size()[0])
    img_path = data['A_paths'][1]
    fake_B = netG(input_A)
    print(f'input a size: {input_A.size(), real_B.size(), fake_B.size()}')
    images_tensor = torch.cat([input_A, real_B, fake_B], axis=-1)
    save_result_pic(images_tensor, img_path[0], "SRtest", output_dir)


