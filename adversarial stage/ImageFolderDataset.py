import torch.utils.data as data
import torchvision.transforms as trans

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

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


def accimage_loader(path):
    print("can't find acc image loader")
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # torchvision 包收录了若干重要的公开数据集、网络模型和计算机视觉中的常用图像变换
    # get_image_backend： 查看载入图片的包的名称
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# define the own imagefolder   from code for torchvision.datasets.folder

class MyImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, dir, transform=None):
        # for target in sorted(os.listdir(dir)):
        #     path = os.path.join(dir, target)
        #     # image_list:
        #     '''
        #     ['E:/derain\\train\\Rain_Heavy',
        #     'E:/derain\\train\\Rain_Light',
        #     'E:/derain\\train\\Rain_Medium']
        #     '''
        #     image_list.append(path)
        # # self.len = len(image_list)
        #self.image_list = image_list
        self.imgs = []
        self.loader = default_loader
        self.transform = transform
        self.root = ""
        #self.classes, self.class_to_index = find_classes(dir)
        self.dir = dir
        #self.images = make_dataset(dir, self.class_to_index)[:4]
        for root, a , images in os.walk(self.dir):
            for i in images:
                self.imgs.append((root, i))
        # self.imgs = self.imgs[:32]
        self.len = len(self.imgs)
        #print(self.len)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        #path = self.image_list[index]
        path = self.imgs[index]
        img = self.loader(path)
        # img = trans.functional.resize(img, [256,256*4])
        if self.transform is not None:
            img = self.transform(img)
            #print(f"img size: {img.size()}")

        return img

    def __len__(self):
        return self.len
