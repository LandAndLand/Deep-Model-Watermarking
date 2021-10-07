import os.path
import torchvision.transforms as trans
import sys
sys.path.append('/home/ay3/houls/Deep-Model-Watermarking/SR attack/')
from data.base_dataset import BaseDataset
from PIL import Image

# loader (callable, optional): A function to load an image given its path.
def pil_loader(path):
    #print(f'image path: {path}')
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img_path = os.path.join(path[0], path[1])
    #print(img_path)
    with open(img_path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            # return img.convert('L')
            # return img

def default_loader(path):
    # torchvision 包收录了若干重要的公开数据集、网络模型和计算机视觉中的常用图像变换
    # get_image_backend： 查看载入图片的包的名称
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# define the own imagefolder   from code for torchvision.datasets.folder

class SRDataset(BaseDataset):
    def __init__(self, opt):
        self.imgs = []
        self.loader = default_loader
        self.root = ""
        #self.classes, self.class_to_index = find_classes(dir)
        #self.dir = os.path.join(opt.dataroot, opt.phase)
        self.dir = opt.dataroot
        #self.images = make_dataset(dir, self.class_to_index)[:4]
        for root, a , images in os.walk(self.dir):
            for i in images:
                self.imgs.append((root, i))
        #print(len(self.imgs))
        #self.imgs = self.imgs[:16]
        self.len = len(self.imgs)
        #print(self.len)
        if opt.input_nc == 1 :
            transform_list = [trans.ToTensor(),
                              trans.Normalize((0.5, ),
                                                   (0.5, ))]
        else:
            transform_list = [trans.ToTensor(),]
                            #   trans.Normalize((0.5,0.5,0.5 ),
                            #                  (0.5,0.5,0.5 ))]
        self.transform = trans.Compose(transform_list)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        #path = self.image_list[index]
        path = self.imgs[index]
        #print(f'SRDataset: {path}')
        img = self.loader(path)
        img = trans.functional.resize(img, [256,512])
        if self.transform is not None:
            img = self.transform(img)
            #print(f"img size: {img.size()}")
        A = img[:, 0:256, 0:256]
        B = img[:, 0:256, 256:] 
        return {'A': A, 'B': B,
                    'A_paths': path, 'B_paths': path}

    def __len__(self):
        return self.len
    def name(self):
        return 'SRDataset'
