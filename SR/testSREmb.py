
import os
import torch
from tqdm.std import tqdm

from SRDataset import SRDataset
from SR_parser import parameter_parser
from tqdm import tqdm
from utils import save_result_pic
from models.HidingRes import HidingRes

opt = parameter_parser()
stage = "IniStage"
dataset_dir = "/home/ay3/houls/watermark_dataset/derain/"
StageDir = os.path.join(dataset_dir, stage)
mode = "train"
# "/home/ay3/houls/watermark_dataset/derain/IniStage/train"
modeStageDir = os.path.join(StageDir, mode)
testDir = os.path.join(dataset_dir, "test")

result_root = "/home/ay3/houls/Deep-Model-Watermarking/result"
result_stage = "derain_flower_Init"
result_time = "2021-09-30-11_20"
result_dir = os.path.join(result_root, result_stage, result_time)
rmodelname = "netR191.pth"
modelpath = os.path.join(result_dir, 'modelrun/outckpts', rmodelname)

# 输入到SR model中的路径
# input_dir = os.path.join(result_dir, input_name)
# input_name = "test"
# input_dir = os.path.join(dataset_dir, "test")
input_dir = os.path.join(result_root, "derain_flower_SR/2021-10-05-21_22", "SRout_two")
output_dir = os.path.join(result_root, "derain_flower_SR/2021-10-05-21_22", "SRout_Rextractone")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
netR = HidingRes(in_c=3, out_c=3)
netR.load_state_dict(torch.load(modelpath))
netR.cuda()
netR.eval()
Rdataset = SRDataset(opt, input_dir)
data_loader = torch.utils.data.DataLoader(
    Rdataset, batch_size=1, shuffle=False, num_workers=int(opt.nThreads))
for i, data in tqdm(enumerate(data_loader)):
    input_A = data['A'].cuda()
    real_B = data['B'].cuda()
    fake_B = data['B1'].cuda()
    # B2 = data['B2'].cuda()
    #print(f'input a size: {input_A.size()}')
    this_batch_size = int(input_A.size()[0])
    img_path = data['A_paths'][1]
    watermark_B1 = netR(fake_B)
    # watermark_B = netR(B2)
    watermark_inputA = netR(input_A)
    images_tensor = torch.cat([input_A, watermark_inputA, real_B, watermark_B1], axis=-1)
    save_result_pic(images_tensor, img_path[0], "testSREmb", output_dir)


