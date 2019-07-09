from __future__ import print_function
import argparse
import os
import time

from skimage import transform as trans
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
import cv2

from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='root path of face images (default: none).')
parser.add_argument('--img_list', default='', type=str, metavar='PATH',
                    help='list of face images for feature extraction (default: none).')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=79077, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')


def main():
    global args
    args = parser.parse_args()

    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else:
        print('Error model type\n')

    model.eval()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    img_list = read_list(args.img_list)
    transform = transforms.Compose([transforms.ToTensor()])
    image_size = (128,128)

    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    src[:, 0] += 8.0
    tform = trans.SimilarityTransform()

    input_data = torch.zeros(1, 1, *image_size)
    for count, (img_name, lmk) in enumerate(img_list):
        img = cv2.imread(os.path.join(args.root_path, img_name), cv2.IMREAD_GRAYSCALE)

        tform.estimate(lmk, src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        img = transform(img)
        input_data[0, :, :, :] = img

        start = time.time()
        if args.cuda:
            input_data = input_data.cuda()
        input_var = torch.autograd.Variable(input_data, volatile=True)
        _, features = model(input_var)
        end = time.time() - start
        print("{}({}/{}). Time: {}".format(os.path.join(args.root_path, img_name), count + 1, len(img_list), end))
        save_feature(args.save_path, img_name, features.data.cpu().numpy()[0])


def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            content = line.strip().split()
            img_name = content[0]
            lmk = np.array([float(x) for x in content[1:-1]], dtype=np.float32)
            lmk = lmk.reshape((5, 2))

            img_list.append((img_name, lmk))
    print('There are {} images..'.format(len(img_list)))
    return img_list


def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir = os.path.dirname(img_path) + '/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.npy'
    np.save(fname, features)


if __name__ == '__main__':
    main()
