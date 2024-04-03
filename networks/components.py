import torch
from torch import nn
import torch.nn.functional as F
from networks.norm import get_norm
from networks.blurpool import get_blurpool

############## Layers ##############

def get_layer(_name):
    return {
        "basic": BasicLayer,
    }[_name]

def get_subsampler(_name, nchannels):
    return {
        "down_max": nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        "down_avg": nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        "down_conv": nn.Sequential(nn.Conv2d(nchannels, nchannels, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU()),
        "down_blurmax": nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), get_blurpool("down_2d")(channels=nchannels, filt_size=3, stride=2)),
        "down_bluravg": get_blurpool("down_2d")(channels=nchannels, filt_size=3, stride=2),
        "up_bilinear": nn.Upsample(scale_factor=2, mode='bilinear'),
        "up_nearest": nn.Upsample(scale_factor=2, mode='nearest'),
        "up_deconv": nn.ConvTranspose2d(nchannels, nchannels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
        "up_blurbilinear": get_blurpool("up")(channels=nchannels),
    }[_name]

class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
    block_names=["ConvBlock", "ConvBlock"],
    kernels=[3,3],
    subsampling="none",
    norms=["evo", "evo"],
    # acts=[nn.LeakyReLU(0.1), nn.LeakyReLU(0.1)]):
    acts=[nn.ReLU(), nn.ReLU()]):
        super(BasicLayer, self).__init__()
        assert len(block_names) == 2
        self.subsampler = get_subsampler(subsampling, in_channels) if subsampling != "none" else None

        self.block1 = get_block(block_names[0])(in_channels, out_channels, kernel_size=kernels[0], norm=norms[0], act=acts[0])
        self.block2 = get_block(block_names[1])(out_channels, out_channels, kernel_size=kernels[1], norm=norms[1], act=acts[1])

    def forward(self, X, _skip_feat=None):
        if self.subsampler is not None:
            X = self.subsampler(X)
        X = self.block1(X)
        if _skip_feat is not None:
            X = X + _skip_feat
        X = self.block2(X)
        return X

class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block_names=["ConvBlock", "ConvBlock"]):
        super(ResLayer, self).__init__()
        assert len(block_names) == 2
        self.block1 = get_block(block_names[0])(in_channels, out_channels, act=nn.ReLU())
        self.block2 = get_block(block_names[1])(out_channels, out_channels, act=nn.ReLU())

    def forward(self, X, _skip_feat=None):
        if _skip_feat is not None:
            X = X + _skip_feat
        residual = X
        X = self.block2(self.block1(X))
        out = X+residual
        return out

############## Blocks ##############

def get_block(block_name):
    return {
        "ConvBlock": ConvBlock,
    }[block_name]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, act=nn.ReLU(), norm="none"):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True))
        if norm != "none":
            layers.append(get_norm(norm)(out_channels))
        else:
            layers.append(act)

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x
