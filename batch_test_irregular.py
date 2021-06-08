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
import cv2

#######################################################   create Irregular Mask        ##############################################################
def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    mask = np.flip(mask, 0)
    mask = np.flip(mask, 1)
    return mask

def generate_stroke_mask(im_size, parts=15, maxVertex=int(20/2), maxLength=int(60/2), maxBrushWidth=int(96/2), maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(np.random.randint(10, parts + 1)):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    return mask

data_name = 'CelebA-HQ'

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
                                               shuffle=True,
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
            if i>5000:
                break

            irr_mask = generate_stroke_mask((256, 256))
            irr_mask = irr_mask[:, :, 0]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            irr_mask = cv2.erode(irr_mask, kernel)
            m = np.array(irr_mask * 255., dtype=np.uint8)
            cv2.imwrite('test_result_irregular_mask/{}mask.png'.format(i), m)

            mask = default_loader('test_result_irregular_mask/{}mask.png'.format(i))
            mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
            x = ground_truth * (1. - mask)
            mask = mask.unsqueeze(dim=0)

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
                vutils.save_image(x[n1, :, :, :], 'test_result_irregular_mask/{}_mask_image.png'.format(n1 + i * args.batch_size), padding=0, normalize=True)
                # vutils.save_image(mask[n1, :, :, :], 'test_result_irregular_mask/{}_mask.png'.format(n1 + i * args.batch_size), padding=0, normalize=True)
                # vutils.save_image(mask[0, :, :], 'test_result_irregular_mask/{}_mask.png'.format(n1 + i * args.batch_size), padding=0, normalize=True)

            for n2 in range(ground_truth.size(0)):
                vutils.save_image(ground_truth[n2, :, :, :], 'test_result_irregular_mask/{}_raw_image.png'.format(n2 + i * args.batch_size), padding=0, normalize=True)

            for n3 in range(inpainted_result.size(0)):
                vutils.save_image(inpainted_result[n3, :, :, :], 'test_result_irregular_mask/{}.png'.format(n3 + i * args.batch_size), padding=0, normalize=True)



if __name__ == '__main__':
    main()
