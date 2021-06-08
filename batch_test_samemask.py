import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list
from data.dataset import Dataset

import numpy as np

import sys
sys.setrecursionlimit(1000000)

data_name = 'CelebA-HQ'
root = '/pubdata/zbw/1,all_paper_code/Design_Anti-forensic/Method_7_conbine_with_pattern_noise/z_Final_plan_code/17,jiuzheng/1,xin/0,new_sacle/1/z_compare_dataset_result/{}/2021Ours_db2/'.format(data_name)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--test_dataroot', type=str, default='/pubdata/zbw/1,all_paper_code/Design_Anti-forensic/Method_7_conbine_with_pattern_noise/4,xin_pytorch/V19_data/{}/test/'.format(data_name))
parser.add_argument('--mask', type=str, default='./examples/center_mask_256.png')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = [0]  # config['gpu_ids']
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

    test_dataset = Dataset(data_path=args.test_dataroot,
                           with_subfolder=True,
                           image_shape=config['image_shape'],
                           random_crop=config['random_crop'])
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
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
    last_model_name = get_model_list(checkpoint_path, "gen", iteration=990000)
    netG.load_state_dict(torch.load(last_model_name))
    model_iteration = int(last_model_name[-11:-3])
    print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

    if cuda:
        netG = nn.parallel.DataParallel(netG, device_ids=device_ids)

    with torch.no_grad():
        netG.eval()
        for i, ground_truth in enumerate(test_loader, 0):
            print(i)


            mask = default_loader(args.mask)
            mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
            x = ground_truth * (1. - mask)
            mask = mask.unsqueeze(dim=0)
            '''

            # Prepare irregular
            bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            # bbox_list = []
            # for ii in range(ground_truth.size()[0]):
            #     bbox_list.append((64, 64, 128, 128))
            # bboxes = torch.tensor(bbox_list, dtype=torch.int64)
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

            ground_truth = (ground_truth + 1.0) / 2.0
            x = (x + 1.0) / 2.0
            inpainted_result = (inpainted_result + 1.0) / 2.0

            for n1 in range(x.size(0)):
                vutils.save_image(x[n1, :, :, :], root+'all/{}_mask_image.png'.format(n1 + i * args.batch_size), padding=0, normalize=True)
                vutils.save_image(mask[n1, :, :, :], root+'all/{}_mask.png'.format(n1 + i * args.batch_size), padding=0, normalize=True)
                vutils.save_image(mask[0, :, :], root+'all/{}_mask.png'.format(n1 + i * args.batch_size), padding=0, normalize=True)
                vutils.save_image(x[n1, :, :, :], root+'mask/{}.png'.format(n1 + i * args.batch_size), padding=0, normalize=True)

            for n2 in range(ground_truth.size(0)):
                vutils.save_image(ground_truth[n2, :, :, :], root+'all/{}_raw_image.png'.format(n2 + i * args.batch_size), padding=0, normalize=True)
                vutils.save_image(ground_truth[n2, :, :, :], root+'ori/{}.png'.format(n2 + i * args.batch_size), padding=0, normalize=True)


            for n3 in range(inpainted_result.size(0)):
                vutils.save_image(inpainted_result[n3, :, :, :], root+'all/{}.png'.format(n3 + i * args.batch_size), padding=0, normalize=True)
                vutils.save_image(inpainted_result[n3, :, :, :], root+'com/{}.png'.format(n3 + i * args.batch_size), padding=0, normalize=True)

            '''
            vutils.save_image(x, 'test/all/{}_mask_image.png'.format(i), padding=0, normalize=True)
            vutils.save_image(ground_truth, 'test/all/{}_raw_image.png'.format(i), padding=0, normalize=True)
            vutils.save_image(inpainted_result, 'test/all/{}.png'.format(i), padding=0)

            vutils.save_image(x, 'test/mask/{}.png'.format(i), padding=0, normalize=True)
            vutils.save_image(ground_truth, 'test/ori/{}.png'.format(i), padding=0, normalize=True)
            vutils.save_image(inpainted_result, 'test/com/{}.png'.format(i), padding=0)
            '''


if __name__ == '__main__':
    main()
