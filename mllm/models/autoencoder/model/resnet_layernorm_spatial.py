import torch
from torch import nn
from mllm.models.autoencoder.layers.layer_norm import LayerNorm2d

norm_map = {
    "ln": LayerNorm2d,
    "bn": LayerNorm2d
}


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm="bn"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = norm_map[norm](out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.norm2 = norm_map[norm](out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.norm3 = norm_map[norm](out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + out
        out = self.relu(out)

        return out


class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None, norm="bn"):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = norm_map[norm](out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.norm2 = norm_map[norm](out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.norm3 = norm_map[norm](out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out = shortcut + out
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes, norm_enc="ln", norm_dec="ln"):
        super(ResNet, self).__init__()

        self.in_channels = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3,
                               bias=False) # 128 x128 x 32
        self.norm1 = norm_map[norm_enc](32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 64 x 64 x 32

        self.dlayer1 = self._make_downlayer(downblock, 32, 2, norm=norm_enc)  # 64 x 64 x 64
        self.dlayer2 = self._make_downlayer(downblock, 128, 2, norm=norm_enc)  # 64 x 64 x 256
        self.dlayer3 = self._make_downlayer(downblock, 256, 2, stride=2, norm=norm_enc)  # 64 x 64 x 512
        self.dlayer4 = self._make_downlayer(downblock, 512, 2, stride=2, norm=norm_enc)  # 32 x 32 x 1024

        self.uplayer1 = self._make_up_block(upblock, 512, 1, stride=1, norm=norm_dec) # 16 x 16
        self.uplayer2 = self._make_up_block(upblock, 64, 2, stride=2, norm=norm_dec) # 32 x 32
        self.uplayer3 = self._make_up_block(upblock, 32, 2, stride=2, norm=norm_dec) # 64 x 64
        self.uplayer4 = self._make_up_block(upblock, 16, 2, stride=2, norm=norm_dec) # 128 x 128
        # self.uplayer5 = self._make_up_block(upblock, 8, 2, stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               16,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            norm_map[norm_dec](16),
        )
        self.uplayer_top = DeconvBottleneck(self.in_channels, 16, 1, 2, upsample) # 256 256

        self.conv1_1 = nn.ConvTranspose2d(16, n_classes, kernel_size=1, stride=1,
                                          bias=False)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1, norm="bn"):
        downsample = None
        norm_ = norm_map[norm]
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_(init_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample, norm))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1, norm="ln"):
        upsample = None
        # expansion = block.expansion
        norm_ = norm_map[norm]
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels*2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                norm_(init_channels*2),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample, norm))
        self.in_channels = init_channels * 2
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 2))
        # layers.append(block(self.in_channels, init_channels, 2, stride, upsample, norm))
        return nn.Sequential(*layers)

    def encode(self, x):
        x_size = x.size()
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dlayer1(x)
        x = self.dlayer2(x)
        x = self.dlayer3(x)
        x = self.dlayer4(x)
        # x = self.dlayer5(x)
        # final_resolution = x.size()
        # x = self.linear(x.reshape(x_size[0], -1))
        # return x.reshape(final_resolution)
        return x

    def decode(self, x, img_size=(256, 256)):
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        # x = self.uplayer5(x)
        x = self.uplayer_top(x)
        x = self.conv1_1(x, output_size=img_size)
        return x

    def decode_prev(self, x, img_size=(256, 256)):
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        # x = self.uplayer5(x)
        x = self.uplayer_top(x)
        return x

    def generate(self, encoder_repr, img_size=(256, 256), ifsigmoid=True):
        # x = encoder_repr.reshape(encoder_repr.shape[0], -1, 8, 8)
        logits = self.decode(encoder_repr, img_size)
        if ifsigmoid:
            return logits.sigmoid()
        else:
            return logits

    def forward(self, x, return_embedding=False, noise_scale=0.):
        x_size = x.size()
        repr = self.encode(x)
        reconstruction = self.decode(repr, x_size)
        if return_embedding:
            return reconstruction, repr.reshape(repr.shape[0], -1)
        return reconstruction


def ResNet50(**kwargs):
    return ResNet(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], 1, **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, DeconvBottleneck, [2,2,2,2], 1, **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 2], 22, **kwargs)
