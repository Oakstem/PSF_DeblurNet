import torch
import torch.nn.functional as F
#import os
#from scipy.io import loadmat
#from torchvision import transforms

class camera_model(torch.nn.Module):
    def __init__(self, weights_shape, if_pad=True):
        super().__init__()
        self.psf = psf(weights_shape, if_pad)

    def forward(self, inputx, weight=None):
        '''
        :param inputx: Batch of N Sequences of images - in shape N,S,C,H,W
        :param weight: PSF Tensor in shape S, C, H, W
        :return: batch of N blurred images using the PSF in shape N,C,H,W
        '''
        # inputs size (N,S,C,H,W), Scene=7, Channels=3
        # weight: (S,C,H,W)
        not_batch = False
        #print(inputx.shape)
        if len(inputx.shape) == 4:
            inputx = torch.unsqueeze(inputx, 0)
            not_batch = True
        #N, S, C, H, W = inputx.shape
        psfed_images = self.psf(inputx, weight) # NSCHW
        simulated_images = psfed_images.mean(1) # NCHW batch
        if not_batch:
            simulated_images = simulated_images.squeeze(0)
        return simulated_images

class psf(torch.nn.Module):
    def __init__(self, weights_shape, if_pad_same=True, padding_mode='replicate'):
        super().__init__()

        assert isinstance(weights_shape, tuple)
        S, C, H, W = weights_shape # Scene_images, Channels, H, W

        print(weights_shape)
        N = S*C
        pad = H//2
        #k= H
        def _pair(p): return (p, p)
        self.stride = _pair(1)
        self.dilation = _pair(1)
        self.groups = N
        self.padding = _pair(pad) if if_pad_same else _pair(0)
        self.register_parameter('bias', None)
        def _reverse_repeat_tuple(t, n): return tuple(x for x in reversed(t) for _ in range(n))
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        self.padding_mode = padding_mode if if_pad_same else 'zeros'
        #super(psf, self).__init__(
        #    in_channels=N, out_channels=N, kernel_size=H, stride=1, padding=pad,
        #    dilation=1, groups=N, bias=False, padding_mode='replicate')
        #self.set_psf_weights(psfs_data)
    def forward(self, input, weight=None):
        '''
        :param input: Batch of N Sequences of images - in shape N,S,C,H,W
        :param weight: PSF Tensor in shape S, C, H, W
        :return: Convolved scene images with the PSF for each Scene image
        '''
        N, S, C, H, W = input.shape
        if not input.is_contiguous():
            input = input.contiguous()
        input = input.view(N, S*C, H, W)

        assert weight is not None, "weights are missing. send in forward call"
        Sk, Ck, Hk, Wk = weight.shape
        weight = weight.view(Sk * Ck, 1, Hk, Wk).to(torch.float32)
        assert S*C == Sk*Ck

        if self.padding_mode != 'zeros':
            pad_input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            out = F.conv2d(pad_input, weight, self.bias, self.stride,
                           (0, 0), self.dilation, self.groups)
        else:
            out = F.conv2d(input, weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
        N2, C2, H2, W2 = out.shape
        psfed_images = out.view(N, S, C, H2, W2)
        return psfed_images

    def set_psf_weights(self, psfs_data):
        del self.weight
        #self.weight = torch.nn.Parameter(psfs_data)#, requires_grad=False)
        #self.weight = psfs_data
        #  and compute psf every run (retain graph issue) ?
        self.register_buffer('weight', psfs_data)
