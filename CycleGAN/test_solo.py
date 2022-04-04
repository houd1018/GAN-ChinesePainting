import argparse
import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from models import Generator
from torch.autograd import Variable
from torchvision.utils import save_image

parser = argparse.ArgumentParser()


parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def letterbox_image(image, size):
    image   = image.convert("RGB")
    iw, ih  = image.size
    w, h    = size
    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image       = image.resize((nw,nh), Image.BICUBIC)
    new_image   = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    if opt.input_nc==1:
        new_image = new_image.convert("L")
    return new_image

def detect_image(image_1, image_2):

    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)

    #---------------------------------------------------#
    #   对输入图像进行不失真的resize
    #---------------------------------------------------#
    image_1 = letterbox_image(image_1,[opt.size,opt.size])
    image_2 = letterbox_image(image_2,[opt.size,opt.size])
    
    #---------------------------------------------------#
    #   对输入图像进行归一化
    #---------------------------------------------------#
    photo_1 = np.asarray(image_1).astype(np.float64) / 255
    photo_2 = np.asarray(image_2).astype(np.float64) / 255

    if opt.input_nc==1:
        photo_1 = np.expand_dims(photo_1, -1)
        photo_2 = np.expand_dims(photo_2, -1)

    with torch.no_grad():
        #---------------------------------------------------#
        #   添加上batch维度，才可以放入网络中预测
        #---------------------------------------------------#
        photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor)
        photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor)
        
        if opt.cuda:
            photo_1 = photo_1.cuda()
            photo_2 = photo_2.cuda()
            netG_A2B.cuda()
            netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()
    
    ###### Testing######

    # Create output dirs if they don't exist
    if not os.path.exists('output_solo/A'):
        os.makedirs('output_solo/A')
    if not os.path.exists('output_solo/B'):
        os.makedirs('output_solo/B')
        
    real_A = Variable(photo_1)
    real_B = Variable(photo_2)

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, 'output_solo/A/A.png')
    save_image(fake_B, 'output_solo/B/B.png')

    sys.stdout.write('Generated images!')

# -----------------------plot-------------------
    plt.subplot(2, 2, 1)
    plt.imshow(np.array(image_1))
    plt.title('Real_A')

    plt.subplot(2, 2, 2)
    Fake_B_root = 'output_solo/B/B.png'
    Fake_B = Image.open(Fake_B_root)
    plt.imshow(np.array(Fake_B))
    plt.title('Fake_B')

    plt.subplot(2, 2, 3)
    plt.imshow(np.array(image_2))
    plt.title('Real_B')

    plt.subplot(2, 2, 4)
    Fake_A_root = 'output_solo/A/A.png'
    Fake_A = Image.open(Fake_A_root)
    plt.imshow(np.array(Fake_A))
    plt.title('Fake_A')

    
    plt.show()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")




if __name__ == "__main__":
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        detect_image(image_1,image_2)