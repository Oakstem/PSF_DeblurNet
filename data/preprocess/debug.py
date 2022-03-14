import numpy as np
import torch


def debug_interp_frames(i1, i2, nb_interp):
    return torch.tensor(np.ones((nb_interp,3,600,700))).double()
