import torch
from torch import nn
from torchvision import models, transforms
import torch.nn.functional as F

import numpy as np

from PIL import Image
from .util import load_image

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        net = models.vgg19(pretrained=True)
        self.feature = nn.Sequential(*(net.features[i] for i in range(36)))
        self.feature.add_module('36: GlobalPooling', nn.AdaptiveMaxPool2d(1))

    def _extract(self, im, train_mode):
        self.training = train_mode
        im = self.image_transform(im).float()
        # N x C x H x W
        im = im.unsqueeze(0)
        feat = self.feature(im)
        feat = feat.view(1, -1).detach().float()
        return F.normalize(feat, p=2, dim=1).numpy()

    def forward(self, train_mode):
        return self._extract(image_path, train_mode)

class ResizeExtractor(FeatureNet):
    def __init__(self, image_size):
        super(ResizeExtractor, self).__init__()
        self.image_size = image_size

    def extract(self, image_path):
        im = load_image(image_path, self.image_size[0], self.image_size[1])
        return self._extract(im, False)

    def forward(self, image_path):
        return self.extract(image_path)

class NoResizeExtractor(FeatureNet):
    def __init__(self):
        super(NoResizeExtractor, self).__init__()

    def extract(self, image_path):
        im = load_image(image_path)
        return self._extract(im, False)

    def forward(self, image_path):
        return self.extract(image_path)
