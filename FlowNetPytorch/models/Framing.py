import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn import ModuleList

from .util import conv, conv_block, deconv, crop_like, Up


class GoWithTheFlownet(nn.Module):
    def __init__(self, device, input_channels=3, batchNorm=True):
        super(GoWithTheFlownet, self).__init__()

        enc_features = [256, 384, 512, 768, 1024, 1280]
        self.encoder = Encoder(input_channels=input_channels, batchNorm=batchNorm, enc_features=enc_features).to(device)

        self.decoder_l = Decoder(device, enc_features)
        self.decoder_r = Decoder(device, enc_features)

    def forward(self, x):
        encs = self.encoder(x)
        left_frames = self.decoder_l(encs)
        right_frames = self.decoder_r(encs)

        return left_frames, right_frames, None, None


class Encoder(nn.Module):
    def __init__(self, enc_features, input_channels=3, batchNorm=True):
        super(Encoder, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv_block(self.batchNorm, input_channels, enc_features[0])
        self.conv2 = conv_block(self.batchNorm, enc_features[0], enc_features[1])
        self.conv3 = conv_block(self.batchNorm, enc_features[1], enc_features[2])
        self.conv4 = conv_block(self.batchNorm, enc_features[2], enc_features[3])
        self.conv5 = conv_block(self.batchNorm, enc_features[3], enc_features[4])
        self.conv6 = conv_block(self.batchNorm, enc_features[4], enc_features[5])

    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)

        return enc1, enc2, enc3, enc4, enc5, enc6


class Decoder(nn.Module):
    def __init__(self, device, enc_features, input_channels=3, levels=6, target_sz=256, batchNorm=True):
        super(Decoder, self).__init__()
        self.target_sz = target_sz

        feature_offs_5 = 20
        feature_offs_4 = 15
        feature_offs_3 = 10
        feature_offs_2 = 5
        feature_offs_1 = 4

        self.decoders = ModuleList()

        self.decoders.append(DenseBlock(num_input_features=2 * enc_features[0], num_layers=4, growth_rate=1).to(device))
        self.decoders.append(DenseBlock(num_input_features=2 * enc_features[1], growth_rate=1).to(device))
        self.decoders.append(DenseBlock(num_input_features=2 * enc_features[2], growth_rate=2).to(device))
        self.decoders.append(DenseBlock(num_input_features=2 * enc_features[3], growth_rate=3).to(device))
        self.decoders.append(DenseBlock(num_input_features=2 * enc_features[4], growth_rate=4).to(device))
        self.decoders.append(DenseBlock(num_input_features=1 * enc_features[5], growth_rate=4).to(device))

        self.dec6_up = deconv(1 * enc_features[5] + feature_offs_5, enc_features[4])
        self.dec5_up = deconv(2 * enc_features[4] + feature_offs_5, enc_features[3])
        self.dec4_up = deconv(2 * enc_features[3] + feature_offs_4, enc_features[2])
        self.dec3_up = deconv(2 * enc_features[2] + feature_offs_3, enc_features[1])
        self.dec2_up = deconv(2 * enc_features[1] + feature_offs_2, enc_features[0])
        self.dec1_up = deconv(2 * enc_features[0] + feature_offs_1, input_channels)

        # self.flow6_up = Up(in_channels=2 * enc_features[5] + feature_offs_5, out_channels=3, ups_factor=25, target_sz=128)
        # self.flow5_up = Up(in_channels=3 * enc_features[4] + feature_offs_5, out_channels=3, ups_factor=14, target_sz=128)
        # self.flow4_up = Up(in_channels=3 * enc_features[3] + feature_offs_4, out_channels=3, ups_factor=7, target_sz=128)
        # self.flow3_up = Up(in_channels=3 * enc_features[2] + feature_offs_3, out_channels=3, ups_factor=3, target_sz=128)
        # self.flow2_up = Up(in_channels=3 * enc_features[1] + feature_offs_2, out_channels=3, ups_factor=4, target_sz=256)
        self.flow1_up = Up(in_channels=2 * enc_features[0] + feature_offs_1, out_channels=3, ups_factor=2,
                           target_sz=target_sz)

        # STN's
        # self.trs = [SpatialTransformer(enc_features[i], level=i).to(device) for i in range(levels)]

    def forward(self, encs):
        # trs = [self.trs[idx](enc) for idx, enc in enumerate(encs)]

        dec6 = self.decoders[5](encs[5])
        dec_up6 = crop_like(self.dec6_up(dec6), encs[4])

        dec5 = self.decoders[4](torch.cat((encs[4], dec_up6), dim=1))
        dec_up5 = crop_like(self.dec5_up(dec5), encs[3])

        dec4 = self.decoders[3](torch.cat((encs[3], dec_up5), dim=1))
        dec_up4 = crop_like(self.dec4_up(dec4), encs[2])

        dec3 = self.decoders[2](torch.cat((encs[2], dec_up4), dim=1))
        dec_up3 = crop_like(self.dec3_up(dec3), encs[1])

        dec2 = self.decoders[1](torch.cat((encs[1], dec_up3), dim=1))
        dec_up2 = crop_like(self.dec2_up(dec2), encs[0])

        dec1 = self.decoders[0](torch.cat((encs[0], dec_up2), dim=1))
        dec_rgb1 = self.flow1_up(dec1)

        return dec_rgb1


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
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
        return self.stn(x)


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    _version = 2

    # The paper states good values for growth rate: 12, 24, 32, 40
    def __init__(self, num_input_features, num_layers=5, bn_size=4, growth_rate=4, drop_rate=0, memory_efficient=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
