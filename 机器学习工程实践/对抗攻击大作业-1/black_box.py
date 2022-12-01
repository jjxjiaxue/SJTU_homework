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

def proj_l2ball(clean_image, adv, epsilon=5):
    adv_noise = (adv - clean_image)
    factor = torch.max(torch.ones([1]).cuda(), torch.norm(adv_noise, p=2) / epsilon)
    adv = clean_image + adv_noise / factor
    adv = torch.clamp(adv, 0, 1)
    return adv

# Copied from LeBA
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel.astype(np.float32)

# Copied from LeBA
def gauss_conv(img, k_size):
    kernel = gkern(k_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    stack_kernel = torch.Tensor(stack_kernel).cuda()
    out = F.conv2d(img, stack_kernel, padding=(k_size-1)//2, groups=3)
    return out

# Motivated and changed by LeBA
def get_trans_advimg(imgs, model2, ba_num=10):
    # TIMI for following iterations in LeBA, similar to attack_black function, but it won't query victim model during iteration
    # Args: ba_num: iteration num in TIMI
    adv_img = imgs.detach().clone()
    adv_img.requires_grad=True
    epsilon = 5
    img_num = imgs.shape[0]
    output = model(imgs)
    pred = output.data.max(1)[1]

    for i in range(ba_num):   
        out = model2(adv_img)
        loss = F.cross_entropy(out, pred)
        loss.backward()
        grad = adv_img.grad.data
        grad = gauss_conv(grad, 9)
        norm2 = torch.norm(grad, p=2)
        norm_grad = grad / norm2
        epsilon_step = epsilon / ba_num
        adv_img = adv_img.data +  epsilon_step * norm_grad
        adv_img.data = proj_l2ball(imgs, adv_img, epsilon=epsilon)
        adv_img = torch.clamp(adv_img, 0, 1)
        adv_img.requires_grad=True

    return gauss_conv(adv_img - imgs, 9).abs()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation code for White Box attack')
    parser.add_argument('-a', '--arch', default='resnet')
    parser.add_argument('-s', '--num_step', default=10000)
    parser.add_argument('-e', '--eps_iter', default=0.004)
    parser.add_argument('-lp', '--label_path', default="attack/old_labels")
    parser.add_argument('-ip', '--image_path', default="attack/images")
    args = parser.parse_args()

    if args.arch == 'resnet':
        model = models.resnet50(pretrained=True).cuda()
        model.eval()
        model = Warp_Resnet(model)
        guide_model = models.vgg16(pretrained=True).cuda()
        guide_model.eval()
        guide_model = Warp_Resnet(guide_model)
    else:
        model = timm.create_model('vit_small_patch16_224', pretrained=True).cuda()
        model.eval()

    epsilon = 5
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

    def prob(model, image, label):
        with torch.no_grad():
            output = model(image).softmax(-1).squeeze()
        if output.argmax().item() != label.item():
            return output[label.item()].item(), True
        return output[label.item()].item(), False
    
    
    eps_iter = args.eps_iter
    dim_tensor = torch.tensor([3 * 224 * 224]).cuda()

    total_acc = list()
    total_query = list()
    for line in tqdm(fr.readlines()):
        torch.manual_seed(2022)
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

        
            
        # Start Simple BA 
        adv = image.clone()
        clean_prob, _ = prob(model, image, label)
        min_prob = clean_prob
        for iq in range(int(args.num_step)):
            noise = (torch.rand_like(image) - 0.5) * 2 * eps_iter

            # # LeBA + (计算prob map方式稍有不同)
            # # 在指导模型梯度强的地方增加更多的噪声
            # if args.arch == 'resnet' and (iq + 1)% 100 == 0:
            #     grad_map = get_trans_advimg(image, guide_model)
            #     norm_noise = torch.norm(noise, p=2)
            #     noise_map = norm_noise * grad_map
            #     noise = noise_map * norm_noise / torch.norm(noise_map, p=2)

            clip_adv = proj_l2ball(original_image, adv + noise)
            tmp_prob, success = prob(model, clip_adv, label)
            if success: 
                adv = clip_adv
                break
            if tmp_prob < min_prob:
                min_prob = tmp_prob
                adv = clip_adv

        check_norm2 = torch.norm(original_image - adv, p=2)
        if check_norm2 > epsilon:
            print("Warning: out of perturbation budget !")
        output = model(adv)
        # save_rgb(original_image, 'clean')
        # save_rgb(adv, 'noise_Black_L2')
        # import ipdb; ipdb.set_trace()
        total_acc.append(success)
        if success: total_query.append(iq)
        # print(iq)
    
    total_acc = np.array(total_acc)
    print(f"ACC: {total_acc.sum() / total_acc.shape[0]}")
    total_query = np.array(total_query)
    print(f"Mean queries: {total_query.mean()}")
