import math

import PIL
import numpy
import torch
import numpy as np
from skimage.io import imread
import torchvision.transforms.functional as TF
from interpolations.sepconv2 import estimate


def create_interpolations(images_list: list, first_place: int, second_place: int):
    if second_place - 1 <= first_place:
        return images_list

    middle_place = first_place + int(math.ceil(second_place - first_place) / 2)
    images_list[middle_place] = estimate(images_list[first_place], images_list[second_place])

    create_interpolations(images_list, first_place, middle_place)
    create_interpolations(images_list, middle_place, second_place)
    return images_list


def gamma_inv(img: np.ndarray, gamm: float = 2.2):
    return torch.pow(img, (1/gamm))


def get_interpolations(first_image: str, second_image: str, num_of_images: int, resize_func, apply_gamma: bool):
    images_list = [None] * (num_of_images + 2)

    im1 = resize_func(TF.to_tensor(imread(first_image)))
    im2 = resize_func(TF.to_tensor(imread(second_image)))
    if apply_gamma:
        im1 = gamma_inv(im1)
        im2 = gamma_inv(im2)

    first_frame = im1.cuda()
    last_frame = im2.cuda()

    images_list[0] = first_frame
    images_list[num_of_images + 1] = last_frame

    # first_frame = torch.FloatTensor(numpy.ascontiguousarray(
    #     numpy.array(PIL.Image.open(first_image))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    # images_list[0] = first_frame
    # last_frame = torch.FloatTensor(numpy.ascontiguousarray(
    #     numpy.array(PIL.Image.open(second_image))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    # images_list[num_of_images + 1] = last_frame

    images_list = create_interpolations(images_list, 0, num_of_images + 1)

    torch_images_list = torch.stack(images_list)

    return torch_images_list
