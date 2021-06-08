import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
import numpy as np
import cv2

from utils.tools import extract_image_patches, flow_to_image, reduce_mean, reduce_sum, default_loader, same_padding

from pytorch_wavelets import DWTForward, DWTInverse

class Generator(nn.Module):
    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.wavelet_basis = config['wavelet_basis']

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.wavelet_basis, self.use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1, LL_x, HL_x_deconv4, LH_x_deconv4, HH_x_deconv4 = self.coarse_generator(x, mask)
        return x_stage1, LL_x, HL_x_deconv4, LH_x_deconv4, HH_x_deconv4

class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, wavelet_basis, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.DWT = DWTForward(J=1, wave=wavelet_basis, mode='periodization').cuda()
        self.IWT = DWTInverse(wave=wavelet_basis, mode='periodization').cuda()

        self.mini_unet1_conv1 = gen_GatedConv(input_dim * 3, cnum * 1, 5, 1, padding=2, rate=1, norm='in', activation='elu')
        self.mini_unet1_conv2 = gen_GatedConv(cnum * 1, cnum * 2, 3, 2, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_conv3 = gen_GatedConv(cnum * 2, cnum * 4, 3, 2, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_conv4 = gen_GatedConv(cnum * 4, cnum * 8, 3, 2, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_conv5 = gen_GatedConv(cnum * 8, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_conv6 = gen_GatedConv(cnum * 8, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_conv7 = gen_GatedConv(cnum * 8, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_deconv1 = gen_conv(cnum * 8 * 2, cnum * 8, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.mini_unet1_deconv1_conv1 = gen_conv(cnum * 8, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_deconv1_conv2 = gen_conv(cnum * 8 * 2, cnum * 8, 3, 1, padding=1, rate=1, norm='in',activation='elu')
        self.mini_unet1_deconv1_gated = gen_GatedConv(cnum * 8 * 2, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_deconv2 = gen_conv(cnum * 4 * 2, cnum * 4, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.mini_unet1_deconv2_conv1 = gen_conv(cnum * 4, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_deconv2_conv2 = gen_conv(cnum * 4 * 2, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_deconv2_gated = gen_GatedConv(cnum * 4 * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_deconv3 = gen_conv(cnum * 2 * 2, cnum * 2, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.mini_unet1_deconv3_conv1 = gen_conv(cnum * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_deconv3_conv2 = gen_conv(cnum * 2 * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_deconv3_gated = gen_GatedConv(cnum * 2 * 2, cnum * 1, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.mini_unet1_conv10 = gen_GatedConv(cnum * 1 * 2, cnum // 2, 3, 1, padding=1, rate=1, norm='none', activation='elu')
        self.mini_unet1_conv11 = gen_GatedConv(cnum // 2, input_dim * 3, 3, 1, padding=1, rate=1, norm='none', activation='none')

        self.conv1 = gen_GatedConv(input_dim, cnum * 1, 5, 1, padding=2, rate=1, norm='in', activation='elu')
        self.conv2 = gen_GatedConv(cnum * 1, cnum * 2, 3, 2, padding=1, rate=1, norm='in', activation='elu')
        self.conv3 = gen_GatedConv(cnum * 2, cnum * 4, 3, 2, padding=1, rate=1, norm='in', activation='elu')
        self.conv4 = gen_GatedConv(cnum * 4, cnum * 8, 3, 2, padding=1, rate=1, norm='in', activation='elu')

        self.conv51_atrous1 = gen_GatedConv(cnum * 8, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv61_atrous1 = gen_GatedConv(cnum * 2, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv61_atrous2 = gen_GatedConv(cnum * 2, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv71_atrous1 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv71_atrous2 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv71_atrous4 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=4, rate=4, norm='in', activation='elu')
        self.conv81_atrous1 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1,  norm='in', activation='elu')
        self.conv81_atrous2 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv81_atrous4 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=4, rate=4, norm='in', activation='elu')
        self.conv81_atrous8 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=8, rate=8,norm='in', activation='elu')
        self.conv91 = gen_GatedConv(cnum * 8, cnum * 8, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')

        self.conv52_atrous1 = gen_GatedConv(cnum * 8, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv62_atrous1 = gen_GatedConv(cnum * 2, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv62_atrous2 = gen_GatedConv(cnum * 2, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv72_atrous1 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1,  norm='in', activation='elu')
        self.conv72_atrous2 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2,  norm='in', activation='elu')
        self.conv72_atrous4 = gen_GatedConv(cnum * 2 * 2, cnum * 2, kernel_size=3, stride=1, padding=4, rate=4, norm='in', activation='elu')
        self.conv82_atrous1 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')
        self.conv82_atrous2 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=2, rate=2, norm='in', activation='elu')
        self.conv82_atrous4 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=4, rate=4, norm='in', activation='elu')
        self.conv82_atrous8 = gen_GatedConv(cnum * 2 * 3, cnum * 2, kernel_size=3, stride=1, padding=8, rate=8, norm='in', activation='elu')
        self.conv92 = gen_GatedConv(cnum * 8, cnum * 8, kernel_size=3, stride=1, padding=1, rate=1, norm='in', activation='elu')

        self.deconv1 = gen_conv(cnum * 8 * 2, cnum * 8, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv1_conv1 = gen_conv(cnum * 8, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv1_conv2 = gen_conv(cnum * 8 * 2, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv1_gated = gen_GatedConv(cnum * 8 * 2, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2 = gen_conv(cnum * 4 * 2, cnum * 4, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv2_conv1 = gen_conv(cnum * 4, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2_conv2 = gen_conv(cnum * 4 * 2, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2_gated = gen_GatedConv(cnum * 4 * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3 = gen_conv(cnum * 2 * 2, cnum * 2, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv3_conv1 = gen_conv(cnum * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3_conv2 = gen_conv(cnum * 2 * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3_gated = gen_GatedConv(cnum * 2 * 2, cnum * 1, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.conv6 = gen_GatedConv(cnum * 1 * 2, cnum // 2, 3, 1, padding=1, rate=1, norm='none', activation='elu')
        self.conv7 = gen_GatedConv(cnum // 2, input_dim, 3, 1, padding=1, rate=1, norm='none', activation='none')

        self.conv8 = gen_GatedConv(input_dim, input_dim, 3, 1, padding=1, rate=1, norm='none', activation='elu')
        self.conv9 = gen_GatedConv(input_dim, input_dim, 3, 1, padding=1, rate=1, norm='none', activation='none')

    def forward(self, x, mask):
        DWT_xl, DWT_xh = self.DWT(x)
        # print(DWT_xl.shape)  # torch.Size([16, 3, 128, 128])
        # print(DWT_xh[0].shape)  # torch.Size([16, 3, 3, 128, 128])
        DWT_x_LL = DWT_xl  # torch.Size([16, 3, 128, 128])
        DWT_x_HL = DWT_xh[0][:, :, 0, :, :]   # torch.Size([16, 3, 128, 128])
        DWT_x_LH = DWT_xh[0][:, :, 1, :, :]   # torch.Size([16, 3, 128, 128])
        DWT_x_HH = DWT_xh[0][:, :, 2, :, :]   # torch.Size([16, 3, 128, 128])
        DWT_x = torch.cat([DWT_x_HL, DWT_x_LH, DWT_x_HH], dim=1)

        waveletCNN_x_conv1 = self.mini_unet1_conv1(DWT_x)
        waveletCNN_x_conv2 = self.mini_unet1_conv2(waveletCNN_x_conv1)
        waveletCNN_x_conv3 = self.mini_unet1_conv3(waveletCNN_x_conv2)
        waveletCNN_x_conv4 = self.mini_unet1_conv4(waveletCNN_x_conv3)
        waveletCNN_x_conv5 = self.mini_unet1_conv5(waveletCNN_x_conv4)
        waveletCNN_x_conv6 = self.mini_unet1_conv6(waveletCNN_x_conv5)

        waveletCNN_x_deconv1_1 = self.mini_unet1_deconv1_conv1(self.mini_unet1_deconv1(torch.cat([waveletCNN_x_conv6, waveletCNN_x_conv4], dim=1)))
        waveletCNN_x_deconv1_2 = self.mini_unet1_deconv1_conv2(torch.cat([F.interpolate(waveletCNN_x_conv6, scale_factor=2, mode='bilinear', align_corners=True), F.interpolate(waveletCNN_x_conv4, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        waveletCNN_x_deconv1 = self.mini_unet1_deconv1_gated(torch.cat([waveletCNN_x_deconv1_1, waveletCNN_x_deconv1_2], dim=1))

        waveletCNN_x_deconv2_1 = self.mini_unet1_deconv2_conv1(self.mini_unet1_deconv2(torch.cat([waveletCNN_x_deconv1, waveletCNN_x_conv3], dim=1)))
        waveletCNN_x_deconv2_2 = self.mini_unet1_deconv2_conv2(torch.cat([F.interpolate(waveletCNN_x_deconv1, scale_factor=2, mode='bilinear', align_corners=True), F.interpolate(waveletCNN_x_conv3, scale_factor=2,  mode='bilinear', align_corners=True)], dim=1))
        waveletCNN_x_deconv2 = self.mini_unet1_deconv2_gated(torch.cat([waveletCNN_x_deconv2_1, waveletCNN_x_deconv2_2], dim=1))

        waveletCNN_x_deconv3_1 = self.mini_unet1_deconv3_conv1(self.mini_unet1_deconv3(torch.cat([waveletCNN_x_deconv2, waveletCNN_x_conv2], dim=1)))
        waveletCNN_x_deconv3_2 = self.mini_unet1_deconv3_conv2(torch.cat([F.interpolate(waveletCNN_x_deconv2, scale_factor=2, mode='bilinear', align_corners=True), F.interpolate(waveletCNN_x_conv2, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        waveletCNN_x_deconv3 = self.mini_unet1_deconv3_gated(torch.cat([waveletCNN_x_deconv3_1, waveletCNN_x_deconv3_2], dim=1))

        waveletCNN_x_conv10 = self.mini_unet1_conv10(torch.cat([waveletCNN_x_deconv3, waveletCNN_x_conv1], dim=1))

        waveletCNN_x_conv11 = self.mini_unet1_conv11(waveletCNN_x_conv10)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        x_conv51_dilated1 = self.conv51_atrous1(x_conv4)
        x_conv61_dilated1 = self.conv61_atrous1(x_conv51_dilated1)
        x_conv61_dilated2 = self.conv61_atrous2(x_conv51_dilated1)
        x_conv71_dilated1 = self.conv71_atrous1(torch.cat([x_conv61_dilated1, x_conv61_dilated2], dim=1))
        x_conv71_dilated2 = self.conv71_atrous2(torch.cat([x_conv61_dilated1, x_conv61_dilated2], dim=1))
        x_conv71_dilated4 = self.conv71_atrous4(torch.cat([x_conv61_dilated1, x_conv61_dilated2], dim=1))
        x_conv81_dilated1 = self.conv81_atrous1(torch.cat([x_conv71_dilated1, x_conv71_dilated2, x_conv71_dilated4], dim=1))
        x_conv81_dilated2 = self.conv81_atrous2(torch.cat([x_conv71_dilated1, x_conv71_dilated2, x_conv71_dilated4], dim=1))
        x_conv81_dilated4 = self.conv81_atrous4(torch.cat([x_conv71_dilated1, x_conv71_dilated2, x_conv71_dilated4], dim=1))
        x_conv81_dilated8 = self.conv81_atrous8(torch.cat([x_conv71_dilated1, x_conv71_dilated2, x_conv71_dilated4], dim=1))
        x_conv91 = self.conv91(torch.cat([x_conv81_dilated1, x_conv81_dilated2, x_conv81_dilated4, x_conv81_dilated8], dim=1))

        x_conv52_dilated1 = self.conv52_atrous1(x_conv91)
        x_conv62_dilated1 = self.conv62_atrous1(x_conv52_dilated1)
        x_conv62_dilated2 = self.conv62_atrous2(x_conv52_dilated1)
        x_conv72_dilated1 = self.conv72_atrous1(torch.cat([x_conv62_dilated1, x_conv62_dilated2], dim=1))
        x_conv72_dilated2 = self.conv72_atrous2(torch.cat([x_conv62_dilated1, x_conv62_dilated2], dim=1))
        x_conv72_dilated4 = self.conv72_atrous4(torch.cat([x_conv62_dilated1, x_conv62_dilated2], dim=1))
        x_conv82_dilated1 = self.conv82_atrous1(torch.cat([x_conv72_dilated1, x_conv72_dilated2, x_conv72_dilated4], dim=1))
        x_conv82_dilated2 = self.conv82_atrous2(torch.cat([x_conv72_dilated1, x_conv72_dilated2, x_conv72_dilated4], dim=1))
        x_conv82_dilated4 = self.conv82_atrous4(torch.cat([x_conv72_dilated1, x_conv72_dilated2, x_conv72_dilated4], dim=1))
        x_conv82_dilated8 = self.conv82_atrous8(torch.cat([x_conv72_dilated1, x_conv72_dilated2, x_conv72_dilated4], dim=1))
        x_conv92 = self.conv92(torch.cat([x_conv82_dilated1, x_conv82_dilated2, x_conv82_dilated4, x_conv82_dilated8], dim=1))

        x_deconv1_1 = self.deconv1_conv1(self.deconv1(torch.cat([x_conv92, x_conv4], dim=1)))
        x_deconv1_2 = self.deconv1_conv2(torch.cat([F.interpolate(x_conv92, scale_factor=2, mode='bilinear', align_corners=True), F.interpolate(x_conv4, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        x_deconv1 = self.deconv1_gated(torch.cat([x_deconv1_1, x_deconv1_2], dim=1))

        x_deconv2_1 = self.deconv2_conv1(self.deconv2(torch.cat([x_deconv1, x_conv3], dim=1)))
        x_deconv2_2 = self.deconv2_conv2(torch.cat([F.interpolate(x_deconv1, scale_factor=2, mode='bilinear', align_corners=True), F.interpolate(x_conv3, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        x_deconv2 = self.deconv2_gated(torch.cat([x_deconv2_1, x_deconv2_2], dim=1))

        x_deconv3_1 = self.deconv3_conv1(self.deconv3(torch.cat([x_deconv2, x_conv2], dim=1)))
        x_deconv3_2 = self.deconv3_conv2(torch.cat([F.interpolate(x_deconv2, scale_factor=2, mode='bilinear', align_corners=True), F.interpolate(x_conv2, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        x_deconv3 = self.deconv3_gated(torch.cat([x_deconv3_1, x_deconv3_2], dim=1))

        x_conv6 = self.conv6(torch.cat([x_deconv3, x_conv1], dim=1))
        x_conv7 = self.conv7(x_conv6)  # torch.Size([16, 3, 256, 256])

        DWT_xl_conv7, DWT_xh_conv7, = self.DWT(x_conv7)
        LL_x = DWT_xl_conv7

        m_batchsize, C, width, height = LL_x.size()
        HL_x_ = waveletCNN_x_conv11[:, 3 * 0:3 * 1, :, :]
        HL_x = HL_x_.view(m_batchsize, C, 1, width, height)
        LH_x_ = waveletCNN_x_conv11[:, 3 * 1:3 * 2, :, :]
        LH_x = LH_x_.view(m_batchsize, C, 1, width, height)
        HH_x_ = waveletCNN_x_conv11[:, 3 * 2:3 * 3, :, :]
        HH_x = HH_x_.view(m_batchsize, C, 1, width, height)

        IWT_x = self.IWT((LL_x, [torch.cat([HL_x, LH_x, HH_x], dim=2)]))

        x_conv8 = self.conv8(IWT_x)
        x_stage1 = self.conv9(x_conv8)  # torch.Size([16, 3, 256, 256])
        x_stage1 = nn.Tanh()(x_stage1)

        return x_stage1, LL_x, HL_x_, LH_x_, HH_x_

class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, norm='none',
             activation='elu', transpose=False):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate, norm=norm,
                       activation=activation, transpose=transpose)

def gen_GatedConv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, activation='elu', norm='none'):
    return GatedConv2d(input_dim, output_dim,
                 kernel_size, stride=stride,
                 conv_padding=padding, dilation=rate,
                 pad_type='zero',
                 activation=activation, norm=norm, sn=False)

def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# -----------------------------------------------
#                Gated ConvBlock
# -----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1,
                 padding=0, conv_padding=0,
                 dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(GatedConv2d, self).__init__()
        self.use_bias = True
        # Initialize the padding scheme
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation, bias=self.use_bias)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation, bias=self.use_bias)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)

        if self.activation:
            x = self.activation(conv) * gated_mask
        else:
            x = conv * gated_mask

        if self.norm:
            x = self.norm(x)

        return x



# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)
