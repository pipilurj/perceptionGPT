from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mllm.models.autoencoder.layers.layer_norm import LayerNorm2d
import math
class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """

    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.LN1 = LayerNorm2d(c_out)
        self.LN2 = LayerNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
        if self.resize:
            self.LN3 = LayerNorm2d(c_out)

    def forward(self, x):
            conv1 = self.LN1(self.conv1(x))
            relu = self.relu(conv1)
            conv2 = self.LN2(self.conv2(relu))
            if self.resize:
                x = self.LN3(self.conv1(x))
            return self.relu(x + conv2)

class VisualAdaptor(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """

    def __init__(self):
        super(VisualAdaptor, self).__init__()
        self.rb1 = ResBlock(1024, 1024, 3, 1, 1, 'encode') # 16 16 16
        self.rb2 = ResBlock(1024, 1024, 3, 1, 1, 'encode') # 16 16 16
        self.conv_final = nn.Conv2d(1024, 1024, 3, 1, 1)

    def forward(self, inputs):
        out = self.rb1(inputs)
        out = self.rb2(out)
        out = self.conv_final(out)
        return out


class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    # 1024 -> 256 -> 64 -> 16 -> 4
    # 16 -> 32 -> 64 -> 128 -> 256
    def __init__(self, dim_in=1024):
        super(Decoder, self).__init__()
        num_layers = int(math.log2(dim_in // 4))//2
        decoder_layers = []
        cur_dim_in = dim_in
        for i in range(num_layers):
            decoder_layers.append(ResBlock(cur_dim_in, cur_dim_in//4, 2, 2, 0, 'decode'))
            decoder_layers.append(ResBlock(cur_dim_in//4, cur_dim_in//4, 3, 1, 1, 'decode'))
            cur_dim_in = cur_dim_in//4
        self.decoder_layers = nn.Sequential(*decoder_layers)
        self.out_conv = nn.ConvTranspose2d(4, 1, 3, 1, 1) # 1 256 256

    def forward(self, inputs):
        out = self.decoder_layers(inputs)
        out_conv = self.out_conv(out)
        return out_conv


class UpsampleBilinear(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.nn.functional.interpolate(x.to(torch.float32), scale_factor=2.0, mode="nearest")
        return x

class DecoderBilinear(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    # 1024 -> 256 -> 64 -> 16 -> 4
    # 16 -> 32 -> 64 -> 128 -> 256
    def __init__(self, dim_in=1024):
        super(DecoderBilinear, self).__init__()
        num_layers = int(math.log2(dim_in // 4))//2
        decoder_layers = []
        cur_dim_in = dim_in
        for i in range(num_layers):
            decoder_layers.append(UpsampleBilinear())
            decoder_layers.append(ResBlock(cur_dim_in, cur_dim_in//4, 3, 1, 1, 'encode'))
            cur_dim_in = cur_dim_in//4
        self.decoder_layers = nn.Sequential(*decoder_layers)
        self.out_conv = nn.ConvTranspose2d(4, 1, 3, 1, 1) # 1 256 256

    def forward(self, inputs):
        out = self.decoder_layers(inputs)
        out_conv = self.out_conv(out)
        return out_conv



class DecoderMoreToken(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    # 1024 -> 256 -> 64 -> 16 -> 4
    # 16 -> 32 -> 64 -> 128 -> 256
    def __init__(self, dim_in=4096):
        super(DecoderMoreToken, self).__init__()
        num_layers = int(math.log2(dim_in // 4))//2
        decoder_layers = []
        cur_dim_in = dim_in
        for i in range(num_layers):
            decoder_layers.append(ResBlock(cur_dim_in, cur_dim_in//4, 2, 2, 0, 'decode'))
            decoder_layers.append(ResBlock(cur_dim_in//4, cur_dim_in//4, 3, 1, 1, 'decode'))
            cur_dim_in = cur_dim_in//4
        self.decoder_layers = nn.Sequential(*decoder_layers)
        self.final_upsample = UpsampleBilinear()
        self.out_conv = nn.Conv2d(4, 1, 3, 1, 1) # 1 256 256

    def forward(self, inputs):
        out = self.decoder_layers(inputs)
        out = self.final_upsample(self.out_conv(out))
        return out