# based on https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
import torch
from torch.nn import functional as F
import torchvision.transforms.functional as F

class ToTensor:

    def __init__(self):
        pass

    def __call__(self, img, target):
        return F.to_tensor(img), F.to_tensor(target)

class Resize:

    def __init__(self):
        pass

    def __call__(self, img, target):
        return F.resize(img, (128,128)), F.resize(target,(128,128))

class RandomBrightness:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            if torch.randn(1) < self.prob:
                image = F.adjust_brightness(image, 0.8)
            else:
                image = F.adjust_brightness(image, 1.2)
        return image, target

class RandomSharpness:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            if torch.randn(1) < self.prob:
                image = F.adjust_sharpness(image, 0.8)
            else:
                image = F.adjust_sharpness(image, 1.2)
        return image, target

class Rotate:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            if torch.randn(1) < self.prob:
                image = F.rotate(image, 20)
                image = F.rotate(target, 20)
            else:
                image = F.rotate(image, 10)
                image = F.rotate(target, 10)
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
