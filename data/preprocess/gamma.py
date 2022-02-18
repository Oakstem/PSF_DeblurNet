import numpy as np
import torch


def apply_gamma_inverse(img: np.ndarray, gamm: float = 2.2):
    return torch.pow(img, (1/gamm))


def apply_gamma(img: np.ndarray, gamm: float = 2.2):
    return torch.pow(img, gamm)
