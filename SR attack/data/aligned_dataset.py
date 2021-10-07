import os.path
import random
import PIL
import torchvision.transforms as transforms
import torch
import sys
sys.path.append('/home/ay3/houls/Deep-Model-Watermarking/SR attack/')
from SR_parser import parameter_parser
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    def __init__(self, opt):
        # super(AlignedDataset,self).__init__(opt)
        self.opt = opt
        self.root = opt.dataroot
        # /home/ay3/houls/watermark_dataset/derain/SrrStage/train or valid
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # 获取dir_AB路径下的所有图片路径
        self.AB_paths = sorted(make_dataset(self.dir_AB))

        #assert(opt.resize_or_crop == 'resize_and_crop')
        # ToTensor()能够把灰度范围从0-255变换到0-1之间
        # 而后面的transform.Normalize()则把0-1变换到(-1,1)
        # 将数据转换为标准正态分布，使模型更容易收敛
        if opt.input_nc == 1 :
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5, ),
                                                   (0.5, ))]
        else:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5 ),
                                                   (0.5,0.5,0.5 ))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]

        # AB = Image.open(AB_path).convert('RGB')
        AB = Image.open(AB_path)
        AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        AB = self.transform(AB)
        # [3, 256, 512]
        #print(AB.size())

        # [3, 512, 256]
        #print(f'AB size: {AB.size(), AB.size(2)}')
        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        A = AB[:, 0:h, 0:w]
        B = AB[:, 0:h, w:] 
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # A = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize]
        # B = AB[:, h_offset:h_offset + self.opt.fineSize,
        #        w + w_offset:w + w_offset + self.opt.fineSize]

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
if __name__ == "__main__":
    args = parameter_parser()
    adataset = AlignedDataset(args)
    dict = adataset.__getitem__(1)
    A = dict['A']
    B = dict['B']
    A_paths = dict['A_paths']
    # new_img_PIL = transforms.ToPILImage()(A).convert('RGB')
    # new_img_PIL.show() 

