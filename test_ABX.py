import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from tensorboardX import SummaryWriter
from utils import *
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--net", type=str, default="net.pth", help='path of log files')
parser.add_argument("--test_differenceL", type=float, default=25, help='difference level used on test set')
parser.add_argument("--test_A", type=str, default="test_N_A", help='path of testing files')
parser.add_argument("--test_B", type=str, default='test_N_B', help='test on denoising or super-resolution')
parser.add_argument("--output", type=str, default="datasets/test_N_Output", help='path of log files')
parser.add_argument("--start_index", type=int, default=0, help="starting index of testing samples")
parser.add_argument("--mode", type=str, default="N", help='Super-resolution (S) or denoise training (N)')

opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    writer = SummaryWriter(opt.output)
    #Upsample_4x = nn.Upsample(scale_factor=4, mode='bilinear')
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.net)))
    model.eval()

    

    # load data info
    print('Loading data info ...\n')
    files_source_A = glob.glob(os.path.join('datasets', opt.test_A, '*.*'))
    files_source_B = glob.glob(os.path.join('datasets', opt.test_B, '*.*'))

    files_source_A.sort()
    files_source_B.sort()
    # process data
    psnr_predict_avg = 0
    psnr_defect_avg = 0
    for f in range(len(files_source_A)):
        # image
        Img_A = cv2.imread(files_source_A[f])
        Img_B = cv2.imread(files_source_B[f])
        if opt.mode ==  'S':
            h, w, c = Img_A.shape
            Img_B = cv2.resize(Img_B, (h, w), interpolation=cv2.INTER_CUBIC)
        Img_A = normalize(np.float32(Img_A[:,:,0]))
        Img_A = np.expand_dims(Img_A, 0)
        Img_A = np.expand_dims(Img_A, 1)

        Img_B = normalize(np.float32(Img_B[:,:,0]))
        Img_B = np.expand_dims(Img_B, 0)
        Img_B = np.expand_dims(Img_B, 1)

        I_A = torch.Tensor(Img_A)
        I_B = torch.Tensor(Img_B)
        I_A, I_B = Variable(I_A.cuda()), Variable(I_B.cuda())
        with torch.no_grad(): # this can save much memory
            output = model(I_B)
            output = torch.clamp(I_B - output, 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported

        
        psnr_predict = batch_PSNR(output, I_A, 1.)
        psnr_predict_avg += psnr_predict
        psnr_defect = batch_PSNR(I_B, I_A, 1.)
        psnr_defect_avg += psnr_defect
        print("%s output psnr_predict %f" % (f, psnr_predict))
        print("%s input psnr_predict %f" % (f, psnr_defect))

        output= output[0,:,:].cpu()
        output= output[0].numpy().astype(np.float32)*255
        cv2.imwrite(os.path.join(opt.output, "%#04d.png" % (f+opt.start_index)), output)
        

    psnr_predict_avg /= len(files_source_A)
    print("\nPSNR on output data %f" % psnr_predict_avg)
    psnr_defect_avg /= len(files_source_A)
    print("\nPSNR on input data %f" % psnr_defect_avg)
    

    I_A = I_A[0,:,:].cpu()
    I_A = I_A[0].numpy().astype(np.float32)
    I_B= I_B[0,:,:].cpu()
    I_B= I_B[0].numpy().astype(np.float32)


    fig = plt.figure()

    ax = plt.subplot("131")
    ax.imshow(I_A, cmap='gray')
    ax.set_title("GT")

    ax = plt.subplot("132")
    ax.imshow(I_B, cmap='gray')
    ax.set_title("defective input")

    ax = plt.subplot("133")
    ax.imshow(output, cmap='gray')
    ax.set_title("Output(DnCNN)")
    plt.show()

if __name__ == "__main__":
    main()
