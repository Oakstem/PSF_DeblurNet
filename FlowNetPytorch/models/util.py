import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# try:
#     from spatial_correlation_sampler import spatial_correlation_sample
# except ImportError as e:
#     import warnings
#     with warnings.catch_warnings():
#         warnings.filterwarnings("default", category=ImportWarning)
#         warnings.warn("failed to load custom correlation module"
#                       "which is needed for FlowNetC", ImportWarning)


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def conv_block(batchNorm, in_planes, out_planes):
    mid_planes = (in_planes+out_planes) // 2
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def warp(self, x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
    # np.save('mask.npy', mask.cpu().data.numpy())
    # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask