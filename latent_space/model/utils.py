import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from bayesian_torch.layers.flipout_layers.linear_flipout import LinearFlipout
import torchvision.models as models
from einops import rearrange, repeat
import math
import random


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, input_channel):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)  # -> 64, 56 x 56
        x = self.layer1(x)  # -> 128, 28 x 28
        x = self.layer2(x)  # -> 256, 14 x 14
        x = self.layer3(x)  # -> 512, 7 x 7
        x = x.view(x.size(0), -1)
        return x


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super(Interpolate, self).__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super(DecoderBlock, self).__init__()
        self.conv1 = self.resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding."""
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

    def resize_conv3x3(self, in_planes, out_planes, scale=1):
        """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
        if scale == 1:
            return self.conv3x3(in_planes, out_planes)
        return nn.Sequential(
            Interpolate(scale_factor=scale), self.conv3x3(in_planes, out_planes)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetDecoder(nn.Module):
    """Resnet in reverse order."""

    def __init__(
        self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False
    ):
        super(ResNetDecoder, self).__init__()

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 64, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(
            64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False
        )

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                self.resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def resize_conv1x1(self, in_planes, out_planes, scale=1):
        """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
        if scale == 1:
            return self.conv1x1(in_planes, out_planes)
        return nn.Sequential(
            Interpolate(scale_factor=scale), self.conv1x1(in_planes, out_planes)
        )

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution."""
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )

    def forward(self, x):
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return x


class ImagePool(object):
    def __init__(self, poolSize):
        super(ImagePool, self).__init__()
        self.poolSize = poolSize
        if poolSize > 0:
            self.num_imgs = 0
            self.images = []

    def Query(self, img):
        # not using lsGAN
        if self.poolSize == 0:
            return img
        if self.num_imgs < self.poolSize:
            # pool is not full
            self.images.append(img)
            self.num_imgs = self.num_imgs + 1
            return img
        else:
            # pool is full, by 50% chance randomly select an image tensor,
            # return it and replace it with the new tensor, by 50% return
            # the newly generated image
            p = random.random()
            if p > 0.5:
                idx = random.randint(0, self.poolSize - 1)
                tmp = self.images[idx]
                self.images[idx] = img
                return tmp
            else:
                return img


class SensorModel(nn.Module):
    """
    the sensor model takes the current raw sensor (low-dimensional sensor)
    and map the raw sensor to latent space

    input -> [batch_size, 1, raw_input]
    output ->  [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, input_size, dim_z):
        super(SensorModel, self).__init__()
        self.dim_z = dim_z
        self.input_size = input_size
        self.num_ensemble = num_ensemble

        self.fc2 = nn.Linear(self.input_size, 128)
        self.fc3 = LinearFlipout(128, 256)
        self.fc4 = LinearFlipout(256, 512)
        self.fc5 = LinearFlipout(512, self.dim_z)
        # self.fc6 = LinearFlipout(256, self.dim_z)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x = repeat(x, "bs dim -> bs k dim", k=self.num_ensemble)
        x = rearrange(x, "bs k dim -> (bs k) dim")

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x, _ = self.fc3(x)
        x = F.leaky_relu(x)
        x, _ = self.fc4(x)
        x = F.leaky_relu(x)
        encoding = x
        obs, _ = self.fc5(x)
        obs = rearrange(
            obs, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        # obs_z = torch.mean(obs, axis=1)
        # obs_z = rearrange(obs_z, "bs (k dim) -> bs k dim", k=1)
        # encoding = rearrange(
        #     encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        # )
        # encoding = torch.mean(encoding, axis=1)
        # encoding = rearrange(encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=1)
        return obs


class miniSensorModel(nn.Module):
    """
    the sensor model takes the current raw sensor (low-dimensional sensor)
    and map the raw sensor to latent space

    input -> [batch_size, 1, raw_input]
    output ->  [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, input_size, dim_z):
        super(miniSensorModel, self).__init__()
        self.dim_z = dim_z
        self.input_size = input_size
        self.num_ensemble = num_ensemble

        self.fc2 = nn.Linear(self.input_size, 128)
        self.fc3 = LinearFlipout(128, 256)
        self.fc6 = LinearFlipout(256, self.dim_z)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x = repeat(x, "bs dim -> bs k dim", k=self.num_ensemble)
        x = rearrange(x, "bs k dim -> (bs k) dim")

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x, _ = self.fc3(x)
        x = F.leaky_relu(x)
        encoding = x
        obs, _ = self.fc6(x)
        obs = rearrange(
            obs, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        # obs_z = torch.mean(obs, axis=1)
        # obs_z = rearrange(obs_z, "bs (k dim) -> bs k dim", k=1)
        # encoding = rearrange(
        #     encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        # )
        # encoding = torch.mean(encoding, axis=1)
        # encoding = rearrange(encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=1)
        return obs


class ImgToLatentModel(nn.Module):
    """
    latent sensor model takes the inputs stacks of images t-n:t-1
    and generate the latent state representations for the transformer
    process model, here we use resnet34 as the basic encoder to project
    down the vision inputs

    images -> [batch, channels, height, width]
    out -> [batch, ensemble, latent_dim_x]
    """

    def __init__(self, num_ensemble, dim_x, input_channel):
        super(ImgToLatentModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        layers = [2, 2, 2, 2]
        self.model = ResNet(ResidualBlock, layers, input_channel)
        self.linear1 = torch.nn.Linear(512 * 7 * 7, 2048)
        self.linear2 = torch.nn.Linear(2048, 1024)
        self.bayes1 = LinearFlipout(in_features=1024, out_features=512)
        self.bayes2 = LinearFlipout(in_features=512, out_features=dim_x)

    def forward(self, images):
        x = self.model(images)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = repeat(x, "bs dim -> bs en dim", en=self.num_ensemble)
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x, _ = self.bayes1(x)
        x = F.relu(x)
        x, _ = self.bayes2(x)
        # x = F.relu(x)
        out = rearrange(x, "(bs en) dim -> bs en dim", en=self.num_ensemble)
        return out


class miniImgToLatentModel(nn.Module):
    """
    latent sensor model takes the inputs stacks of images t-n:t-1
    and generate the latent state representations for the transformer
    process model, here we use resnet34 as the basic encoder to project
    down the vision inputs

    images -> [batch, channels, height, width]
    out -> [batch, ensemble, latent_dim_x]
    """

    def __init__(self, num_ensemble, dim_x, input_channel):
        super(miniImgToLatentModel, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, 16, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=0.1),
        )
        self.linear1 = torch.nn.Linear(64 * 7 * 7, 512)
        self.bayes1 = LinearFlipout(in_features=512, out_features=256)
        self.bayes2 = LinearFlipout(in_features=256, out_features=dim_x)

    def forward(self, images):
        x = images
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = repeat(x, "bs dim -> bs en dim", en=self.num_ensemble)
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x, _ = self.bayes1(x)
        x = F.leaky_relu(x)
        x, _ = self.bayes2(x)
        out = rearrange(x, "(bs en) dim -> bs en dim", en=self.num_ensemble)
        return out


class ImgToLatentModel_Baseline(nn.Module):
    """
    latent sensor model takes the inputs stacks of images t-n:t-1
    and generate the latent state representations for the transformer
    process model, here we use resnet34 as the basic encoder to project
    down the vision inputs

    images -> [batch, channels, height, width]
    out -> [batch, ensemble, latent_dim_x]
    """

    def __init__(self, num_ensemble, dim_x, input_channel):
        super(ImgToLatentModel_Baseline, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        layers = [2, 2, 2, 2]
        self.model = ResNet(ResidualBlock, layers, input_channel)
        self.linear1 = torch.nn.Linear(512 * 7 * 7, 2048)
        self.linear2 = torch.nn.Linear(2048, 1024)
        self.bayes1 = LinearFlipout(in_features=1024, out_features=32)
        self.bayes2 = LinearFlipout(in_features=32, out_features=dim_x)

    def forward(self, images):
        batch_size = images.shape[0]
        x = self.model(images)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = repeat(x, "bs dim -> bs en dim", en=self.num_ensemble)
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x, _ = self.bayes1(x)
        x = F.leaky_relu(x)
        encoding = x
        x, _ = self.bayes2(x)
        out = rearrange(x, "(bs en) dim -> bs en dim", en=self.num_ensemble)
        encoding = rearrange(encoding, "(bs en) dim -> bs en dim", en=self.num_ensemble)
        encoding = torch.mean(encoding, axis=1)
        encoding = rearrange(encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=1)
        return out, encoding


class SensorModel_Baseline(nn.Module):
    """
    the sensor model takes the current raw sensor (low-dimensional sensor)
    and map the raw sensor to latent space

    input -> [batch_size, 1, raw_input]
    output ->  [batch_size, num_ensemble, dim_z]
    """

    def __init__(self, num_ensemble, input_size, dim_z):
        super(SensorModel_Baseline, self).__init__()
        self.dim_z = dim_z
        self.input_size = input_size
        self.num_ensemble = num_ensemble

        self.fc2 = nn.Linear(self.input_size, 128)
        self.fc3 = LinearFlipout(128, 256)
        self.fc4 = LinearFlipout(256, 512)
        self.fc5 = LinearFlipout(512, 64)
        self.fc6 = LinearFlipout(64, 32)
        self.fc7 = LinearFlipout(32, self.dim_z)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x = repeat(x, "bs dim -> bs k dim", k=self.num_ensemble)
        x = rearrange(x, "bs k dim -> (bs k) dim")

        x = self.fc2(x)
        x = F.leaky_relu(x)
        x, _ = self.fc3(x)
        x = F.leaky_relu(x)
        x, _ = self.fc4(x)
        x = F.leaky_relu(x)
        x, _ = self.fc5(x)
        x = F.leaky_relu(x)
        x, _ = self.fc6(x)
        x = F.leaky_relu(x)
        encoding = x
        obs, _ = self.fc7(x)
        obs = rearrange(
            obs, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        # obs_z = torch.mean(obs, axis=1)
        # obs_z = rearrange(obs_z, "bs (k dim) -> bs k dim", k=1)
        encoding = rearrange(
            encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        encoding = torch.mean(encoding, axis=1)
        encoding = rearrange(encoding, "(bs k) dim -> bs k dim", bs=batch_size, k=1)
        return obs, encoding


class ImageRecover(nn.Module):
    """
    image recover module is the decoder model which reconstruct the predicted image
    from the latent space, here we assume the latent vector is a good representation
    and we use a couple upconv layer to recover the image

    latent_vec -> [batch, latent_dim_x]
    out -> [batch, channels, height, width]
    """

    def __init__(self, num_ensemble, dim_x):
        super(ImageRecover, self).__init__()
        self.num_ensemble = num_ensemble
        self.dim_x = dim_x
        self.decoder = ResNetDecoder(
            DecoderBlock,
            [2, 2, 2, 2],
            self.dim_x,
            input_height=224,
            first_conv=False,
            maxpool1=False,
        )

    def forward(self, x):
        img = self.decoder(x)
        return img


class AuxiliaryModel(nn.Module):
    """
    auxiliary model is used to support the image generator is able to generate the
    images that is actually making sense and the state/observation can be learned
    from the constructed image.

    img -> [batch, channels, height, width]
    out -> [batch_size, 1, raw_input]
    """

    def __init__(self, input_size):
        super(AuxiliaryModel, self).__init__()
        self.input_size = input_size
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=0.1),
        )
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, self.input_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        out = rearrange(x, "(bs k) dim -> bs k dim", k=1)
        return out


class Discriminator(nn.Module):
    """
    auxiliary model is used to support the image generator is able to generate the
    images that is actually making sense and the state/observation can be learned
    from the constructed image.

    img -> [batch, channels, height, width]
    out -> [batch_size, p] where p = [0,1]
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=0.1),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=0.1),
        )
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        out = torch.sigmoid(x)
        return out


class DecoderModel(nn.Module):
    """
    input -> [batch_size, ensemble, dim_x]
    output ->  [batch_size, num_ensemble, dim_gt]
    """

    def __init__(self, num_ensemble, dim_x, dim_gt):
        super(DecoderModel, self).__init__()
        self.dim_x = dim_x
        self.dim_gt = dim_gt
        self.num_ensemble = num_ensemble

        self.fc2 = nn.Linear(dim_x, 256)
        self.fc3 = LinearFlipout(256, 128)
        self.fc4 = LinearFlipout(128, 32)
        self.fc5 = LinearFlipout(32, self.dim_gt)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x, _ = self.fc3(x)
        x = F.leaky_relu(x)
        x, _ = self.fc4(x)
        x = F.leaky_relu(x)
        state, _ = self.fc5(x)

        state = rearrange(
            state, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        m_state = torch.mean(state, axis=1)
        m_state = rearrange(m_state, "bs (k dim) -> bs k dim", k=1)
        return state, m_state


class miniDecoderModel(nn.Module):
    """
    input -> [batch_size, ensemble, dim_x]
    output ->  [batch_size, num_ensemble, dim_gt]
    """

    def __init__(self, num_ensemble, dim_x, dim_gt):
        super(miniDecoderModel, self).__init__()
        self.dim_x = dim_x
        self.dim_gt = dim_gt
        self.num_ensemble = num_ensemble

        self.fc2 = nn.Linear(dim_x, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, self.dim_gt)

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "bs k dim -> (bs k) dim")
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        state = self.fc5(x)

        state = rearrange(
            state, "(bs k) dim -> bs k dim", bs=batch_size, k=self.num_ensemble
        )
        m_state = torch.mean(state, axis=1)
        m_state = rearrange(m_state, "bs (k dim) -> bs k dim", k=1)
        return state, m_state
