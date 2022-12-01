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

def save_rgb(image, name):
    image = image.detach().cpu().squeeze().numpy()
    image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(image).save(f'{name}.png')

class Warp_Resnet(torch.nn.Module):
    def __init__(self, resnet) -> None:
        super().__init__()
        self.model = resnet
        self.mean = torch.tensor([0.485,0.456,0.406]).cuda().view([3,1,1])
        self.std = torch.tensor([0.229,0.224,0.225]).cuda().view([3,1,1])
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation code for White Box attack')
    parser.add_argument('-a', '--arch', default='resnet')
    parser.add_argument('-d', '--decay', default=1)
    parser.add_argument('-s', '--num_step', default=80)
    parser.add_argument('-e', '--eps_iter', default=0.04)
    parser.add_argument('-lp', '--label_path', default="attack/old_labels")
    parser.add_argument('-ip', '--image_path', default="attack/images")
    args = parser.parse_args()

    if args.arch == 'resnet':
        model = models.resnet50(pretrained=True).cuda()
        model.eval()
        model = Warp_Resnet(model)
    else:
        model = timm.create_model('vit_small_patch16_224', pretrained=True).cuda()
        model.eval()

    epsilon = 0.3
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

    dim_tensor = torch.tensor([3 * 224 * 224]).cuda()

    total_acc = list()
    for line in tqdm(fr.readlines()):
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
        
        # Start PGD L2 Attack
        # Random Start (PGD)
        x = image
        x += torch.rand(x.shape).cuda() * (epsilon / torch.sqrt(dim_tensor))
        x = Variable(x, requires_grad=True)

        for i in range(num_step):
            
            output = model(x)

            loss = loss_function(output, pred)
            loss.backward()

            grad = x.grad.data
        
            norm2 = torch.norm(grad, p=2)
            norm_grad = grad / norm2

            epsilon_step = float(args.eps_iter)

            # step lambda decay
            if args.decay and i >= (num_step // 2): epsilon_step = epsilon_step / 2

            adv = x.data +  epsilon_step * norm_grad

            # Clip adversarial noise out of L2 Norm
            adv_noise = (adv - original_image)
            factor = torch.max(torch.ones([1]).cuda(), torch.norm(adv_noise, p=2) / epsilon)
            adv = original_image + adv_noise / factor

            # Clip 0 - 1
            adv = torch.clamp(adv, 0, 1)

            x = Variable(adv, requires_grad=True)

        adv = x
        check_norm2 = torch.norm(original_image - adv, p=2)
        assert(check_norm2 < epsilon + 1e-3)
        # save_rgb(original_image, 'clean')
        # save_rgb(adv, 'noise_L2')
        # import ipdb; ipdb.set_trace()
        output = model(adv)
        pred = output.data.max(1)[1].item()
        total_acc.append(label.item() == pred)
    
    total_acc = np.array(total_acc)
    print(f"ACC: {1 - total_acc.sum() / total_acc.shape[0]}")
