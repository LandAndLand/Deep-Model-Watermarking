import os
import random
import shutil

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
        self.derain_imgs = self.derain_imgs[:12100]
        #print(self.derain_imgs)
        #print(len(self.derain_imgs))
    def copyimg(self, tarpath, *imgs):
        #print(type(imgs))
        # 这里的imgs传过来时是一个元组
        imgs = imgs[0]
        #print(type(imgs))
        index = 0
        for arr in imgs:
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
    process = DataProcess(ori_path = derain_path)
    traindata = process.derain_imgs[:6000]
    gandata = process.derain_imgs[6000:12000]
    testdata = process.derain_imgs[12000:]
    process.copyimg(trainpath, traindata)
    process.copyimg(validpath, gandata)
    process.copyimg(testpath, testdata)