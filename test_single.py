import os
import random
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list

from pytorch_wavelets import DWTForward, DWTInverse

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--image', type=str, default='examples/raw_imgHQ02823.png')
parser.add_argument('--mask', type=str, default='examples/center_mask_256.png')
parser.add_argument('--number', type=str, default='')
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

import cv2
import numpy as np

def mask_return_index(mask): # 除缺失区域的“纵向方向”和“上下方向”外，各随机选一块patch
    mask = mask[0]
    mask = np.transpose(mask, [1, 2, 0])

    # mask: (h,w,1) numpy
    image_height = mask.shape[0]
    image_weight = mask.shape[1]

    for i in range(image_height):
        for j in range(image_weight):
            if(mask[i,j] != 0.):
                mask[i,j] = 1.

    pixel_1_index = np.where(mask == 1.)
    pixel_1_index_zhong = pixel_1_index[0]
    pixel_1_index_heng = pixel_1_index[1]
    # print(pixel_1_index_zhong[0], pixel_1_index_heng[0])  # 左上角坐标
    # print(pixel_1_index_zhong[-1], pixel_1_index_heng[-1])  # 右下角坐标

    zhong_length = int(pixel_1_index_zhong[-1] - pixel_1_index_zhong[0] + 1)
    heng_length = int(pixel_1_index_heng[-1] - pixel_1_index_heng[0] + 1)

    return mask, pixel_1_index_zhong[0], pixel_1_index_heng[0], zhong_length, heng_length

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    DWT = DWTForward(J=1, wave='db2', mode='periodization')
    IWT = DWTInverse(wave='db2', mode='periodization')

    # CUDA configuration
    cuda = config['cuda']
    device_ids = [6]  # config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True
        DWT.cuda()
        IWT.cuda()

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    print("Configuration: {}".format(config))

    try:  # for unexpected error logging
        with torch.no_grad():   # enter no grad context
            if is_image_file(args.image):
                if args.mask and is_image_file(args.mask):
                    # Test a single masked image with a given mask
                    x = default_loader(args.image)
                    mask = default_loader(args.mask)
                    if x.size[0] > config['image_shape'][0] or x.size[1] > config['image_shape'][1]:
                        x = transforms.RandomCrop(min(x.size[0],x.size[1]))(x)
                        x = transforms.Resize(config['image_shape'][:-1])(x)
                    elif x.size[0] == config['image_shape'][0] and x.size[1] == config['image_shape'][1]:
                        pass
                    else:
                        x = transforms.Resize(config['image_shape'][:-1])(x)
                        x = transforms.RandomCrop(min(x.size[0], x.size[1]))(x)

                    import cv2
                    import numpy as np
                    x = transforms.ToTensor()(x)
                    mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
                    x = normalize(x)
                    vutils.save_image(x.unsqueeze(dim=0), 'raw_image.png', padding=0,normalize=True)
                    vutils.save_image(x.unsqueeze(dim=0), 'test_result/' + args.number + '_raw_image.png', padding=0, normalize=True)
                    gt = x.unsqueeze(dim=0)
                    x = x * (1. - mask)
                    x = x.unsqueeze(dim=0)
                    missing_image = x
                    mask = mask.unsqueeze(dim=0)

                    vutils.save_image(x, 'mask_image.png', padding=0, normalize=True)
                    vutils.save_image(x, 'test_result/'+args.number+'_mask_image.png', padding=0, normalize=True)
                elif args.mask:
                    raise TypeError("{} is not an image file.".format(args.mask))
                else:
                    # Test a single ground-truth image with a random mask
                    ground_truth = default_loader(args.image)

                    if ground_truth.size[0] > config['image_shape'][0] or ground_truth.size[1] > config['image_shape'][1]:
                        ground_truth = transforms.RandomCrop(min(ground_truth.size[0], ground_truth.size[1]))(ground_truth)
                        ground_truth = transforms.Resize(config['image_shape'][:-1])(ground_truth)
                    elif ground_truth.size[0] == config['image_shape'][0] and ground_truth.size[1] == config['image_shape'][1]:
                        pass
                    else:
                        ground_truth = transforms.Resize(config['image_shape'][:-1])(ground_truth)
                        ground_truth = transforms.RandomCrop(min(ground_truth.size[0], ground_truth.size[1]))(ground_truth)

                    ground_truth = transforms.ToTensor()(ground_truth)
                    ground_truth = normalize(ground_truth)
                    ground_truth = ground_truth.unsqueeze(dim=0)
                    gt = ground_truth
                    bboxes = random_bbox(config, batch_size=ground_truth.size(0))
                    x, mask = mask_image(ground_truth, bboxes, config)
                    missing_image = x
                    import cv2
                    import numpy as np
                    vutils.save_image(x, 'test_result_ramdon_mask/'+args.number+'_mask_image.png', padding=0, normalize=True)
                    vutils.save_image(mask, 'test_result_ramdon_mask/'+args.number+'mask.png', padding=0, normalize=True)
                    vutils.save_image(ground_truth, 'test_result_ramdon_mask/' + args.number + '_raw_image.png', padding=0, normalize=True)

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
                    x = x.cuda()
                    mask = mask.cuda()
                    gt = gt.cuda()
                    missing_image = missing_image.cuda()

                # Inference
                x1, _LL_x_deconv4, _HL_x_deconv4, _LH_x_deconv4, _HH_x_deconv4 = netG(x, mask)
                inpainted_result = x1 * mask + x * (1. - mask)

                vutils.save_image(inpainted_result, args.output, padding=0, normalize=True)
                print("Saved the inpainted result to {}".format(args.output))
                if args.flow:
                    vutils.save_image(offset_flow, args.flow, padding=0, normalize=True)
                    print("Saved offset flow to {}".format(args.flow))
            else:
                raise TypeError("{} is not an image file.".format)
        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    main()
