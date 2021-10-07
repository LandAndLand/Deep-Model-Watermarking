import argparse
import time

from google.protobuf.symbol_database import Default


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetdir', type=str, default="/home/ay3/houls/watermark_dataset/derain", help='path to all derain images')
    parser.add_argument('--dataroot', type=str, default="/home/ay3/houls/watermark_dataset/derain/SrrStage",
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--initroot', type=str, default="/home/ay3/houls/watermark_dataset/derain/IniStage",
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--outnum', type=int, default=4,
                        help="how many photos can be concated for output")
    parser.add_argument('--num', type=int, default=3,
                        help="how many photos can be concated for input")
    parser.add_argument('--whichH', default="191")
    parser.add_argument('--imageSize',type=int, default=256)
    parser.add_argument('--batchSize', type=int,
                        default=2, help='input batch size')
    parser.add_argument('--loadSizeX', type=int, default=256,
                        help='scale images to this size')
    parser.add_argument('--loadSizeY', type=int, default=256,
                        help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=256,
                        help='then crop to this size')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64,
                        help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64,
                        help='# of discrim filters in first conv layer')
    parser.add_argument('--which_model_netD', type=str,
                        default='basic', help='selects model to use for netD')
    parser.add_argument('--which_model_netG', type=str,
                        default='resnet_9blocks', help='selects model to use for netG')
    parser.add_argument('--learn_residual', action='store_true',
                        help='if specified, model would learn only the residual to the input')
    parser.add_argument('--gan_type', type=str, default='wgan-gp',
                        help='wgan-gp : Wasserstein GAN with Gradient Penalty, lsgan : Least Sqaures GAN, gan : Vanilla GAN')
    parser.add_argument('--n_layers_D', type=int, default=3,
                        help='only used if which_model_netD==n_layers')
    parser.add_argument('--gpu_ids', type=list,
                        default=[0, 1, 2], help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--name', type=str, default='saved_model',
                        help='name of the experiment. It decides where to store samples and models')
    # aligned是加载 两张图片在一起的数据集
    # unaligned是加载A B两张图片不在一起的数据集
    # single是用于测试时的数据集
    parser.add_argument('--dataset_mode', type=str, default='myself',
                        help='chooses how datasets are loaded. [unaligned | aligned | single]')
    parser.add_argument('--model', type=str, default='test',
                        help='chooses which model to use. pix2pix, test, content_gan')
    parser.add_argument('--ganloss', type=str, default='yes',
                        help='chooses gan or nogan')
    parser.add_argument('--which_direction', type=str,
                        default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--nThreads', default=1, type=int,
                        help='# threads for loading data')
    parser.add_argument('--checkpoints_dir', type=str,
                        default=f'./result/derain_flower_SR/2021-09-27-15_50/checkpoints', help='models are saved here')
    parser.add_argument('--datetime', type=str,
                        default=f'2021-10-05-21_22', help='models are saved here time')
    parser.add_argument(
        '--dir', type=str, default='./result/derain_flower_SR', help='SR processes are saved')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--serial_batches', default=True,
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--display_winsize', type=int,
                        default=256,  help='display window size')
    parser.add_argument('--display_id', type=int, default=1,
                        help='window id of the web display')
    parser.add_argument('--display_port', type=int,
                        default=8097, help='visdom port of the web display')
    parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--no_dropout', action='store_true',
                        help='no dropout for the generator')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                        help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    parser.add_argument('--no_flip', type=bool, default=True,
                        help='if specified, do not flip the images for data augmentation')

    parser.add_argument('--display_freq', type=int, default=100,
                        help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000,
                        help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--which_epoch', type=str, default='219',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--niter', type=int, default=150,
                        help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=150,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam')
    parser.add_argument('--lambda_A', type=float, default=100.0,
                        help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0,
                        help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
    parser.add_argument('--pool_size', type=int, default=50,
                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('--no_html', action='store_true',
                        help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--ntest', type=int,
                        default=float("inf"), help='# of test examples.')
    parser.add_argument('--results_dir', type=str, default="/home/ay3/houls/watermark_dataset/derain/SrrStage/srrout-B2",
                        help='saves Srr output (B'') results here.')
    parser.add_argument('--aspect_ratio', type=float,
                        default=1.0, help='aspect ratio of result images')
    parser.add_argument('--phase', type=str, default='test',
                        help='train, val, test, etc')
    parser.add_argument('--how_many', type=int, default=10000,
                        help='how many test images to run')
    parser.add_argument('--isTrain', type=bool, default=False, help="is train")
    parser.add_argument(
        '--outroot', default='/home/ay3/houls/Deep-Model-Watermarking/result/derain_flower_SR', help='folder to output images')
    args, unknown = parser.parse_known_args()
    return args
