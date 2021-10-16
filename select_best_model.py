import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import models
import inspect

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list, local_patch
from data.dataset import Dataset

from DISTS import DISTS

import numpy
import numpy as np
import math

import sys
sys.setrecursionlimit(1000000)

class LPIPSvgg(torch.nn.Module):
    def __init__(self, channels=3):
        # Refer to https://github.com/richzhang/PerceptualSimilarity

        assert channels == 3
        super(LPIPSvgg, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [64, 128, 256, 512, 512]
        self.weights = torch.load(os.path.abspath(os.path.join(inspect.getfile(LPIPSvgg), '..', 'weights/LPIPSvgg.pt')))
        self.weights = list(self.weights.items())

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        for k in range(len(outs)):
            outs[k] = F.normalize(outs[k])
        return outs

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        score = 0
        for k in range(len(self.chns)):
            score = score + (self.weights[k][1] * (feats0[k] - feats1[k]) ** 2).mean([2, 3]).sum(1)
        if as_loss:
            return score.mean()
        else:
            return score


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        # a = torch.hann_window(5,periodic=False)
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out + 1e-12).sqrt()

def psnr(img1, img2):
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# real = scipy.misc.imread('dataset/test/test/006_im.png').astype(numpy.float32)[32:32+64,32:32+64,:]
# recon = scipy.misc.imread('out.png').astype(numpy.float32)[32:32+64,32:32+64,:]
# print(psnr(real,recon))

def SSIMnp(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def mae(img1, img2):
    mae = np.mean(abs(img1 - img2))
    return mae



parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--mask', type=str, default='./examples/center_mask_256.png')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

def main(select_dataroot, iteration):
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = [1]  # config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    test_dataset = Dataset(data_path=select_dataroot,
                           with_subfolder=True,
                           image_shape=config['image_shape'],
                           random_crop=config['random_crop'])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=config['num_workers'])
    print("Configuration: {}".format(config))

    # Set checkpoint path
    if not args.checkpoint_path:
        checkpoint_path = os.path.join('checkpoints',
                                       config['dataset_name'],
                                       config['mask_type'] + '_' + config['expname'])
    else:
        checkpoint_path = args.checkpoint_path

    # Define the trainer
    netG = Generator(config['netG'], cuda, device_ids)
    # Resume weight
    last_model_name = get_model_list(checkpoint_path, "gen", iteration=iteration)
    netG.load_state_dict(torch.load(last_model_name))
    model_iteration = int(last_model_name[-11:-3])
    print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LPIPSmodel = LPIPSvgg().to(device_)
    DISTSmodel = DISTS().to(device_)

    if cuda:
        netG = nn.parallel.DataParallel(netG, device_ids=device_ids)

    d_mae = 0
    d_psnr = 0
    d_ssim = 0
    d_LPIPS = 0
    d_DISTS = 0

    with torch.no_grad():
        netG.eval()
        for i, ground_truth in enumerate(test_loader, 0):
            mask = default_loader(args.mask)
            mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
            x = ground_truth * (1. - mask)
            mask = mask.unsqueeze(dim=0)

            '''
            # Prepare irregular
            # bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            bbox_list = []
            for ii in range(ground_truth.size()[0]):
                bbox_list.append((64, 64, 128, 128))
            bboxes = torch.tensor(bbox_list, dtype=torch.int64)
            x, mask, spatial_discounting_mask_tensor = mask_image(ground_truth, bboxes, config)
            mask = mask.squeeze(dim=0)
            '''

            if cuda:
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()

            # Inference
            x1, _LL_x_deconv4, _HL_x_deconv4, _LH_x_deconv4, _HH_x_deconv4 = netG(x, mask)
            inpainted_result = x1 * mask + ground_truth * (1. - mask)

            inpainted_result = (inpainted_result + 1.0) / 2.0
            ground_truth = (ground_truth + 1.0) / 2.0

            local_inpainted_result = inpainted_result[:, :, 64:64+128, 64:64+128]
            local_ground_truth = ground_truth[:, :, 64:64+128, 64:64+128]

            real_np = np.uint8(ground_truth.cpu().numpy().astype(numpy.float32)[0] * 255.)
            fake_np = np.uint8(inpainted_result.cpu().numpy().astype(numpy.float32)[0] * 255.)
            d_mae += mae(real_np, fake_np)
            d_psnr += psnr(real_np, fake_np)
            d_ssim += SSIMnp(real_np, fake_np)

            d_LPIPS += LPIPSmodel(local_inpainted_result, local_ground_truth, as_loss=False).item()
            d_DISTS += DISTSmodel(local_inpainted_result, local_ground_truth, as_loss=False).item()

        d_mae = float(d_mae / len(test_loader))
        d_psnr = float(d_psnr / len(test_loader))
        d_ssim = float(d_ssim / len(test_loader))
        d_LPIPS = float(d_LPIPS / len(test_loader))
        d_DISTS = float(d_DISTS / len(test_loader))

    return d_mae, d_psnr, d_ssim, d_LPIPS, d_DISTS


if __name__ == '__main__':
    val_inital_errG = 1000
    pre_iteration = 5000

    select_data = 'val'
    select_dataroot = '/data/zbw/a/V19/CelebA-HQ/{}/'.format(select_data)
    for iteration in range(5000, 1000000+1, 5000):
        d_mae, d_psnr, d_ssim, d_LPIPS, d_DISTS = main(select_dataroot, iteration)

        message = 'iteration:{}   d_mae:{}   d_psnr:{}    d_ssim:{}   d_LPIPS:{}   d_DISTS:{}'.format(iteration, d_mae, d_psnr, d_ssim, d_LPIPS, d_DISTS)

        if (d_DISTS < val_inital_errG):
            val_inital_errG = d_DISTS
            pre_iteration = iteration

        if select_data == 'test':
            with open("checkpoints/CelebA-HQ/select_best_model_{}.txt".format(select_data), "a") as f:
                f.write(message)
                f.write('\r\n')
                f.write("best_model:{}".format(pre_iteration))
                f.write('\r\n')

                print(message)
                print("\033[1;31m best_model:{}\033[0m".format(pre_iteration))
        else:
            with open("checkpoints/CelebA-HQ/select_best_model_{}.txt".format(select_data), "a") as f:
                f.write(message)
                f.write('\r\n')
                f.write("best_model:{}".format(pre_iteration))
                f.write('\r\n')

                print(message)
                print("\033[1;31m best_model:{}\033[0m".format(pre_iteration))


