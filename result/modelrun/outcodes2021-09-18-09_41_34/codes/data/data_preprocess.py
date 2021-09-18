import os
import random
import shutil

from tqdm import tqdm

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

derain_path = "E:\\derain"        
trainpath = 'E:\\watermark_dataset\\derain\\train'
validpath = 'E:\\watermark_dataset\\derain\\valid'
testpath = 'E:\\watermark_dataset\\derain\\test'
mkdirs(trainpath)
mkdirs(validpath)
mkdirs(testpath)

class DataProcess:
    def __init__(self, ori_path, mode="train") -> None:
        self.derain_imgs = []
        for root, subdir, files in os.walk(os.path.join(ori_path, mode)):
            if len(files)>0:
                print(root)
                #print(files[:5])
                for img in files:
                    #print(img)
                    self.derain_imgs.append((root, img))
        
        random.shuffle(self.derain_imgs)
        if mode!='test':
            self.derain_imgs = self.derain_imgs[:6000]
        else:
            self.derain_imgs = self.derain_imgs[:100]

        #print(self.derain_imgs)
        #print(len(self.derain_imgs))
    def copyimg(self, tarpath):
        index = 0
        for arr in tqdm(self.derain_imgs):
            #print(arr)
            #print(arr[0])
            #print(arr[1])
            old_path = os.path.join(arr[0], arr[1])
            #print(old_path)
            #print(ori_path)
            new_path = os.path.join(tarpath, f"{str(index)}.jpg")
            #print(new_path)
            shutil.copyfile(old_path, new_path)
            index+=1
            #print(index)

if __name__ == "__main__":
    for m in ['train', 'valid', 'test']:
        process = DataProcess(ori_path = derain_path, mode=m)
        process.copyimg(trainpath)
        process.copyimg(validpath)
        process.copyimg(testpath)