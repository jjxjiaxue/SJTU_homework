from re import X
import torchvision.models as models
from torch.autograd import Variable
import os
import argparse

import torchvision.models as models
import timm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
import torchvision.transforms as trans
from PIL import Image
from tqdm import tqdm

from copy import deepcopy


class Warp_Resnet(torch.nn.Module):
    def __init__(self, resnet) -> None:
        super().__init__()
        self.model = resnet
        self.mean = torch.tensor([0.485,0.456,0.406]).cuda().view([3,1,1])
        self.std = torch.tensor([0.229,0.224,0.225]).cuda().view([3,1,1])
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

def save_rgb(image, name):
    image = image.detach().cpu().squeeze().numpy()
    image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(image).save(f'{name}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation code for White Box attack')
    parser.add_argument('-a', '--arch', default='resnet')
    parser.add_argument('-d', '--decay', default=1)
    parser.add_argument('-s', '--num_step', default=80)
    parser.add_argument('-e', '--eps_iter', default=0.1)
    parser.add_argument('-m', '--select_mode', default=3)
    parser.add_argument('-lp', '--label_path', default="attack/old_labels")
    parser.add_argument('-ip', '--image_path', default="attack/images")
    args = parser.parse_args()

    args.select_mode = int(args.select_mode)
    print(args.select_mode)

    if args.arch == 'resnet':
        model = models.resnet50(pretrained=True).cuda()
        model.eval()
        model = Warp_Resnet(model)
    else:
        model = timm.create_model('vit_small_patch16_224', pretrained=True).cuda()
        model.eval()

    num_step = int(args.num_step)
    loss_function = nn.CrossEntropyLoss()
    fr = open(args.label_path)

    filepath=args.image_path
    filenames=os.listdir(filepath)

    preprocess=trans.Compose([
        trans.Resize(256),
        trans.CenterCrop(224),
        trans.ToTensor()
    ])

    mean_kernel = torch.ones(16,16).unsqueeze(0).unsqueeze(0).cuda() / 16 / 16
    mean_kernel = mean_kernel.repeat(1, 3, 1, 1) / 3

    total_acc = list()
    for idx, line in enumerate(tqdm(fr.readlines())):
        # if idx > 50: break
        item = line.split()
        #print(list)
        original_image =Image.open(os.path.join(filepath, item[0]))

        #print(image)
        image = preprocess(original_image).cuda().unsqueeze(0)

        output = model(image)
        pred = output.data.max(1)[1]

        # copy raw image
        original_image = deepcopy(image)

        # Drop the misclassified sample
        label = torch.tensor([int(item[1])]).cuda()
        if pred.item() != label: continue

        # maximum_margin = [max_y:max_y+16, max_x:max_x+16]
        if args.select_mode == 0:
            # Find the patch position with greatest gradient information
            x = image
            x = Variable(x, requires_grad=True)
            output = model(x)
            loss = loss_function(output, pred)
            loss.backward()
            grad_abs = x.grad.data.abs()
            mean_grad = F.conv2d(grad_abs, mean_kernel).squeeze()
            # mean_grad: shape [209, 209]
            mean_grad_argmax = mean_grad.argmax().item()
            max_y, max_x = mean_grad_argmax // 209, mean_grad_argmax % 209
    
        # Find the mask which influenes most using grid masks
        elif args.select_mode == 1:
            ret_lists = list()
            for i in range(0, 224 - 16, 4):
                # 52
                input_lists = list()
                for j in range(0, 224 - 16, 4):
                    temp = deepcopy(image)
                    temp[:, :, i:i+16, j:j+16] = 0
                    input_lists.append(temp)
                input_lists = torch.cat(input_lists, 0)
                with torch.no_grad():
                    ret = model(input_lists)
                ret_lists.append(ret.max(-1)[0])
            ret_max = torch.stack(ret_lists).argmin().item()
            max_y, max_x = ret_max // 52, ret_max % 52

        elif args.select_mode == 2:
            max_y, max_x = 0, 0
  
        else:
            assert(args.select_mode == 3)
            # Center patch
            max_y, max_x = 105, 105

        mask = torch.zeros_like(image)
        mask[:, :, max_y:max_y+16, max_x:max_x+16] = 1
        
        # Start PGD Attack
        # No Random Start (PGD)
        x = image
        # x += torch.rand(x.shape).cuda() * mask
        # save_rgb(x, 'noise')
        x = Variable(x, requires_grad=True)

        for i in range(num_step):
            
            output = model(x)

            loss = loss_function(output, pred)
            loss.backward()

            grad = x.grad.data

            norm2 = torch.norm(grad, p=2)
            norm_grad = grad / norm2

            sign_grad = grad.sign()
            epsilon_step = float(args.eps_iter)

            # step lambda decay
            if args.decay and i >= (num_step // 2): epsilon_step = epsilon_step / 2
            x = x.data + epsilon_step * sign_grad * mask

            # Clip 0 - 1
            x = torch.clamp(x, 0, 1)

            x = Variable(x, requires_grad=True)

        adv = original_image * (1 - mask) + x * mask

        # save_rgb(original_image, 'clean')
        # save_rgb(adv, 'noise_patch')
        # import ipdb; ipdb.set_trace()
        output = model(adv)
        pred = output.data.max(1)[1].item()
        total_acc.append(label.item() == pred)
        # print(label.item() == pred)
        
    
    total_acc = np.array(total_acc)
    print(f"ACC: {1 - total_acc.sum() / total_acc.shape[0]}")
