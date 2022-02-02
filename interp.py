import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from math import ceil
from skimage.io import imread
import re

from argparse import Namespace

from torch.backends import cudnn
import math
from ABME.utils import warp
from ABME.model import SBMENet, ABMRNet, SynthesisNet

from utils import read_pfm


def load_abme_ckpt():
    cudnn.benchmark = True
    args = Namespace(DDP=False)


    SBMNet = SBMENet()
    ABMNet = ABMRNet()
    SynNet = SynthesisNet(args)

    SBMNet.load_state_dict(torch.load('ABME/Best/SBME_ckpt.pth', map_location='cuda'))
    ABMNet.load_state_dict(torch.load('ABME/Best/ABMR_ckpt.pth', map_location='cuda'))
    SynNet.load_state_dict(torch.load('ABME/Best/SynNet_ckpt.pth', map_location='cuda'))

    for param in SBMNet.parameters():
        param.requires_grad = False
    for param in ABMNet.parameters():
        param.requires_grad = False
    for param in SynNet.parameters():
        param.requires_grad = False

    SBMNet.cuda()
    SBMNet.eval()
    ABMNet.cuda()
    ABMNet.eval()
    SynNet.cuda()
    SynNet.eval()

    return SBMNet, ABMNet, SynNet


def get_frame(sbmnet: torch.nn.Module, abmnet: torch.nn.Module, synnet: torch.nn.Module,
              frame1: torch.Tensor, frame3: torch.Tensor, oflow: torch.Tensor = None):


    with torch.no_grad():

        H = frame1.shape[2]
        W = frame1.shape[3]

        # 4K video requires GPU memory of more than 24GB. We recommend crop it into 4 regions with some margin.
        if H < 512:
            divisor = 64.
            D_factor = 1.
        else:
            divisor = 128.
            D_factor = 0.5

        H_ = int(ceil(H / divisor) * divisor * D_factor)
        W_ = int(ceil(W / divisor) * divisor * D_factor)

        frame1_ = F.interpolate(frame1, (H_, W_), mode='bicubic')
        frame3_ = F.interpolate(frame3, (H_, W_), mode='bicubic')

        if oflow is None:
            SBM = sbmnet(torch.cat((frame1_, frame3_), dim=1))[0]
            SBM_= F.interpolate(SBM, scale_factor=4, mode='bilinear') * 20.0
        else:
            SBM_= F.interpolate(oflow, (H_, W_), mode='bilinear') / 6

        frame2_1, Mask2_1 = warp(frame1_, SBM_ * (-1),  return_mask=True)
        frame2_3, Mask2_3 = warp(frame3_, SBM_       ,  return_mask=True)

        frame2_Anchor_ = (frame2_1 + frame2_3) / 2
        frame2_Anchor = frame2_Anchor_ + 0.5 * (frame2_3 * (1-Mask2_1) + frame2_1 * (1-Mask2_3))

        Z  = F.l1_loss(frame2_3, frame2_1, reduction='none').mean(1, True)
        Z_ = F.interpolate(Z, scale_factor=0.25, mode='bilinear') * (-20.0)

        ABM_bw, _ = abmnet(torch.cat((frame2_Anchor, frame1_), dim=1), SBM*(-1), Z_.exp())
        ABM_fw, _ = abmnet(torch.cat((frame2_Anchor, frame3_), dim=1), SBM, Z_.exp())

        SBM_     = F.interpolate(SBM, (H, W), mode='bilinear')   * 20.0
        ABM_fw   = F.interpolate(ABM_fw, (H, W), mode='bilinear') * 20.0
        ABM_bw   = F.interpolate(ABM_bw, (H, W), mode='bilinear') * 20.0

        SBM_[:, 0, :, :] *= W / float(W_)
        SBM_[:, 1, :, :] *= H / float(H_)
        ABM_fw[:, 0, :, :] *= W / float(W_)
        ABM_fw[:, 1, :, :] *= H / float(H_)
        ABM_bw[:, 0, :, :] *= W / float(W_)
        ABM_bw[:, 1, :, :] *= H / float(H_)

        divisor = 8.
        H_ = int(ceil(H / divisor) * divisor)
        W_ = int(ceil(W / divisor) * divisor)

        Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw), dim=1)

        Syn_inputs = F.interpolate(Syn_inputs, (H_,W_), mode='bilinear')
        Syn_inputs[:, 6, :, :] *= float(W_) / W
        Syn_inputs[:, 7, :, :] *= float(H_) / H
        Syn_inputs[:, 8, :, :] *= float(W_) / W
        Syn_inputs[:, 9, :, :] *= float(H_) / H
        Syn_inputs[:, 10, :, :] *= float(W_) / W
        Syn_inputs[:, 11, :, :] *= float(H_) / H

        result = synnet(Syn_inputs)

        result = F.interpolate(result, (H,W), mode='bicubic')

        return result


def create_interpolations(sbmnet: torch.nn.Module, abmnet: torch.nn.Module, synnet: torch.nn.Module,
                          images_list: list, first_place: int, second_place: int, oflow: torch.tensor):
    if second_place - 1 <= first_place:
        return images_list

    middle_place = first_place + int(math.ceil(second_place - first_place) / 2)
    if first_place == 0 and second_place == len(images_list):
        images_list[middle_place] = get_frame(sbmnet, abmnet, synnet,
                                              images_list[first_place], images_list[second_place], oflow)
    else:
        images_list[middle_place] = get_frame(sbmnet, abmnet, synnet,
                                              images_list[first_place], images_list[second_place])
    create_interpolations(sbmnet, abmnet, synnet, images_list, first_place, middle_place, oflow)
    create_interpolations(sbmnet, abmnet, synnet, images_list, middle_place, second_place, oflow)

    return images_list


def gamma_inv(img: np.ndarray, gamm: float = 2.2):
    return torch.pow(img, (1/gamm))


def gamma(img: np.ndarray, gamm: float = 2.2):
    return img**(gamm)


def get_interpolations(first_image: str, second_image: str,
                       num_of_images: int, resize_func, apply_gamma: bool = True, opticalf: str=None):
    images_list = [None] * (num_of_images + 2)

    im1 = imread(first_image)
    im2 = imread(second_image)
    if apply_gamma:
        im1 = gamma_inv(im1)
        im2 = gamma_inv(im2)
    im1 = resize_func(TF.to_tensor(im1))
    im2 = resize_func(TF.to_tensor(im2))
    first_frame = im1.unsqueeze(0).cuda()
    last_frame = im2.unsqueeze(0).cuda()

    images_list[0] = first_frame
    images_list[num_of_images + 1] = last_frame

    sbmnet, abmnet, synnet = load_abme_ckpt()
    if opticalf is not None:
        oflow_ten = -read_pfm(opticalf)[0][..., :2]
        oflow = torch.tensor(oflow_ten).to("cuda")
    else:
        oflow = None
    images_list = create_interpolations(sbmnet, abmnet, synnet, images_list, 0, num_of_images + 1, oflow)
    torch_images_list = torch.cat(images_list)
    # for idx in range(torch_images_list.shape[0]):
    #     if apply_gamma:
    # torch_images_list[idx] = gamma(torch_images_list[idx])

    return torch_images_list
