import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--IniStageRoot',default="/home/ay3/houls/watermark_dataset/derain/IniStage")
    parser.add_argument('--dataroot', type=str, default="/home/ay3/houls/watermark_dataset/derain/SrrStage/after-watermark",
                        help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--modeldir',default="result/derain_flower_Init/modelrun/outckpts/2021-09-24-21_50_35")
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=2,
                        help='input batch size')
    parser.add_argument('--outnum', type=int, default=4,
                        help="how many photos can be concated for output")
    parser.add_argument('--num', type=int, default=4,
                        help="how many photos can be concated for input")
    parser.add_argument('--imageSize', type=int, default=256,
                        help='the number of frames 256/512')
    parser.add_argument('--niter', type=int, default=200,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate, default=0.001')
    parser.add_argument('--decay_round', type=int, default=10,
                        help='learning rate decay 0.5 each decay_round')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--whichG', default="219")
    parser.add_argument('--Rnet', default='',
                        help="path to Revealnet (to continue training)")
    parser.add_argument('--trainpics', default='./result/derain_flower_Adver/pics/trainpics/',
                        help='folder to output training images')
    parser.add_argument('--validationpics', default='./result/derain_flower_Adver/pics/validationpics/',
                        help='folder to output validation images')
    parser.add_argument('--testPics', default='./result/derain_flower_Adver/pics/testPics/',
                        help='folder to output test images')
    parser.add_argument('--runfolder', default='./result/derain_flower_Adver/modelrun/runfolder/',
                        help='for tensorboard to save writter content for draw')
    parser.add_argument('--outckpts', default='./result/derain_flower_Adver/modelrun/outckpts/',
                        help='folder to output checkpoints')
    parser.add_argument('--outlogs', default='./result/derain_flower_Adver/modelrun/outlogs/',
                        help='folder to output images')
    parser.add_argument('--outcodes', default='./result/derain_flower_Adver/modelrun/outcodes/',
                        help='folder to save the experiment codes')

    parser.add_argument('--remark', default='', help='comment')
    parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
    #parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
    parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
    parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')

    #read secret image
    parser.add_argument('--secret', type=str, default='flower',
                        help='secret folder')
    parser.add_argument('--datasets', type=str, default='/home/ay3/houls/watermark_dataset/derain')
    parser.add_argument('--name', default="InitStage",
                        help='InitStage | SrrStage')
    #parser.add_argument('--part', default="train", help='train | valid')
    #hyperparameter of loss

    parser.add_argument('--beta', type=float, default=1,
                        help='hyper parameter of beta :secret_reveal err')
    parser.add_argument('--betamse', type=float, default=10000,
                        help='larger loss for training')
    parser.add_argument('--betagan', type=float, default=1,
                        help='hyper parameter of beta :gans weight')
    parser.add_argument('--betagans', type=float, default=0.01,
                        help='hyper parameter of beta :gans weight')
    parser.add_argument('--betapix', type=float, default=1.0,
                        help='hyper parameter of beta :pixel_loss weight')

    parser.add_argument('--betacons', type=float, default=1,
                        help='hyper parameter of beta: consist_loss')
    parser.add_argument('--betaclean', type=float, default=1,
                        help='hyper parameter of beta: clean_loss')
    parser.add_argument('--betacleanA', type=float, default=0,
                        help='hyper parameter of beta: clean_loss')
    parser.add_argument('--betacleanB', type=float, default=1,
                        help='hyper parameter of beta: clean_loss')


    parser.add_argument('--betavgg', type=float, default=1,
                        help='hyper parameter of beta: vgg_loss')
    parser.add_argument('--betapsnr', type=float, default=0,
                        help='hyper parameter of beta: psnr_loss')
    args, unknown = parser.parse_known_args()
    return args