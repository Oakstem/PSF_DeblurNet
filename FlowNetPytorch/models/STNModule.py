""" A plug and play Spatial Transformer Module in Pytorch """
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer
    as proposed in the Jaderberg paper.
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator
    3. A roi pooled module.

    The current implementation uses a very small convolutional net with
    2 convolutional layers and 2 fully connected layers. Backends
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map.
    """

    def __init__(self, in_channels, level, kernel_size=3, use_dropout=True):
        super(SpatialTransformer, self).__init__()
        # self._h, self._w = spatial_dims
        features_dict = {0: 8192, 1: 2048, 2: 512, 3: 128, 4: 32, 5: 32}
        self._in_ch = in_channels
        self._ksize = kernel_size
        self.dropout = use_dropout
        self.fc_features = features_dict[level]
        # localization net
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1,
                               bias=False)  # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)

        # self.fc1 = nn.Linear(32 * 4 * 4, 1024)
        self.fc1 = nn.Linear(self.fc_features, 1024)
        self.fc2 = nn.Linear(1024, 6)

        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        """
        Forward pass of the STN module.
        x -> input feature map
        """
        batch_images = x
        # get spatial dims
        h, w = x.shape[2:]
        x = F.relu(self.conv1(x.detach()))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        # if x.shape[-1] > 3:
        #     x = F.max_pool2d(x, 6)
        # else:
        #     x = F.max_pool2d(x, 3)
        # Uncomment & replace maxpool for img size 256
        if x.shape[-1] > 1:
            x = F.max_pool2d(x, 2)

        # print("Pre view size:{}".format(x.size()))
        # div = np.prod(x.shape) // batch_images.shape[0]
        # x = x.view(-1, 32 * 4 * 4)
        x = x.view(-1, self.fc_features)
        if self.dropout:
            # x = F.dropout(F.linear(div, 1024), p=0.5)
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x)  # params [Nx6]

        x = x.view(-1, 2, 3)  # change it to the 2x3 matrix
        # print(x.size())
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, h, w)))
        assert (affine_grid_points.size(0) == batch_images.size(
            0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points)
        # print("rois found to be of size:{}".format(rois.size()))
        return rois


class SpatialTransformerSm(nn.Module):
    def __init__(self):
        super(SpatialTransformerSm, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)





