import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        # 模型保存地址
        # self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.save_dir = opt.checkpoints_dir

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        #print(f'model save path: {save_path}')
        # network.load_state_dict(torch.load(save_path))
        # /home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-09-27-22_30/checkpoints/7_netG.pth
        network.load_state_dict(torch.load('/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR/2021-09-27-22_30/checkpoints/114_netG.pth'))


    def update_learning_rate():
        pass
