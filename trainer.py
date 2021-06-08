import os
import torch
import torch.nn as nn
from torch import autograd
from model.networks import Generator, LocalDis, GlobalDis

from utils.tools import get_model_list, local_patch, spatial_discounting_mask, my_discounting_mask
from utils.logger import get_logger

from pytorch_wavelets import DWTForward, DWTInverse
import torch.nn.functional as F

logger = get_logger()

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.add_module('vgg', VGG19())
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))

        self.DWT = DWTForward(J=1, wave=self.config['netG']['wavelet_basis'], mode='periodization').cuda()
        self.IWT = DWTInverse(wave=self.config['netG']['wavelet_basis'], mode='periodization').cuda()

        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.localD.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])
            self.vgg.to(self.device_ids[0])

    def forward(self, x, bboxes, masks, ground_truth, spatial_discounting_mask_tensor, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        x1, LL_x_deconv4, HL_x_deconv4, LH_x_deconv4, HH_x_deconv4 = self.netG(x, masks)
        DWT_xl_gt, DWT_xh_gt = self.DWT(ground_truth)
        LL_x_gt = DWT_xl_gt
        HL_x_gt = DWT_xh_gt[0][:, :, 0, :, :]
        LH_x_gt = DWT_xh_gt[0][:, :, 1, :, :]
        HH_x_gt = DWT_xh_gt[0][:, :, 2, :, :]
        spatial_discounting_mask_tensor_128 = F.interpolate(spatial_discounting_mask_tensor, scale_factor=1 / 2, mode='bilinear', align_corners=True)
        spatial_discounting_mask_tensor_64 = F.interpolate(spatial_discounting_mask_tensor, scale_factor=1 / 4, mode='bilinear', align_corners=True)
        spatial_discounting_mask_tensor_32 = F.interpolate(spatial_discounting_mask_tensor, scale_factor=1 / 8, mode='bilinear', align_corners=True)
        spatial_discounting_mask_tensor_16 = F.interpolate(spatial_discounting_mask_tensor, scale_factor=1 / 16, mode='bilinear', align_corners=True)
        local_patch_gt = local_patch(ground_truth, bboxes)
        x1_inpaint = x1 * masks + x * (1. - masks)
        local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)

        x1_inpaint_vgg = self.vgg((x1_inpaint + 1.0) / 2.0)
        ground_truth_vgg = self.vgg((ground_truth + 1.0) / 2.0)

        # D part
        # wgan d loss
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(self.localD, local_patch_gt, local_patch_x1_inpaint.detach())
        global_real_pred, global_fake_pred = self.dis_forward(self.globalD, ground_truth, x1_inpaint.detach())
        losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
            torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # gradients penalty loss
        local_penalty = self.calc_gradient_penalty(self.localD, local_patch_gt, local_patch_x1_inpaint.detach())
        global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x1_inpaint.detach())
        losses['wgan_gp'] = local_penalty + global_penalty

        # G part
        if compute_loss_g:
            losses['Context'] = l1_loss(x1_inpaint * spatial_discounting_mask_tensor, ground_truth * spatial_discounting_mask_tensor)
            losses['DWT'] = l1_loss(LL_x_deconv4 * spatial_discounting_mask_tensor_128, LL_x_gt * spatial_discounting_mask_tensor_128) + \
                            l1_loss(HL_x_deconv4 * spatial_discounting_mask_tensor_128, HL_x_gt * spatial_discounting_mask_tensor_128) + \
                            l1_loss(LH_x_deconv4 * spatial_discounting_mask_tensor_128, LH_x_gt * spatial_discounting_mask_tensor_128) + \
                            l1_loss(HH_x_deconv4 * spatial_discounting_mask_tensor_128, HH_x_gt * spatial_discounting_mask_tensor_128)
            losses['Perceptual'] = l1_loss(x1_inpaint_vgg['relu1_1'] * spatial_discounting_mask_tensor, ground_truth_vgg['relu1_1'] * spatial_discounting_mask_tensor) + \
                                   l1_loss(x1_inpaint_vgg['relu2_1'] * spatial_discounting_mask_tensor_128, ground_truth_vgg['relu2_1'] * spatial_discounting_mask_tensor_128) + \
                                   l1_loss(x1_inpaint_vgg['relu3_1'] * spatial_discounting_mask_tensor_64, ground_truth_vgg['relu3_1'] * spatial_discounting_mask_tensor_64) + \
                                   l1_loss(x1_inpaint_vgg['relu4_1'] * spatial_discounting_mask_tensor_32, ground_truth_vgg['relu4_1'] * spatial_discounting_mask_tensor_32) + \
                                   l1_loss(x1_inpaint_vgg['relu5_1'] * spatial_discounting_mask_tensor_16, ground_truth_vgg['relu5_1'] * spatial_discounting_mask_tensor_16)
            losses['Style'] = l1_loss(self.compute_gram(x1_inpaint_vgg['relu2_2'] * spatial_discounting_mask_tensor_128), self.compute_gram(ground_truth_vgg['relu2_2'] * spatial_discounting_mask_tensor_128)) + \
                              l1_loss(self.compute_gram(x1_inpaint_vgg['relu3_4'] * spatial_discounting_mask_tensor_64), self.compute_gram(ground_truth_vgg['relu3_4'] * spatial_discounting_mask_tensor_64)) + \
                              l1_loss(self.compute_gram(x1_inpaint_vgg['relu4_4'] * spatial_discounting_mask_tensor_32), self.compute_gram(ground_truth_vgg['relu4_4'] * spatial_discounting_mask_tensor_32)) + \
                              l1_loss(self.compute_gram(x1_inpaint_vgg['relu5_2'] * spatial_discounting_mask_tensor_16), self.compute_gram(ground_truth_vgg['relu5_2'] * spatial_discounting_mask_tensor_16))

            # wgan g loss
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward( self.localD, local_patch_gt, local_patch_x1_inpaint)
            global_real_pred, global_fake_pred = self.dis_forward(self.globalD, ground_truth, x1_inpaint)
            losses['wgan_g'] = - torch.mean(local_patch_fake_pred) - torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

        return losses, x1_inpaint

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1 = self.netG(x, masks)
        x1_inpaint = x1 * masks + x * (1. - masks)

        return x1_inpaint

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save({'localD': self.localD.state_dict(),
                    'globalD': self.globalD.state_dict()}, dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(),
                    'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        self.netG.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])

        if not test:
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            state_dict = torch.load(last_model_name)
            self.localD.load_state_dict(state_dict['localD'])
            self.globalD.load_state_dict(state_dict['globalD'])
            # Load optimizers
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            self.optimizer_g.load_state_dict(state_dict['gen'])

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

import torchvision.models as models
# Assume input range is [0, 1]
class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
