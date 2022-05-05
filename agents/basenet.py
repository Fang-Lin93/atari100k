
import torch
from torch import nn
from torch.nn import functional as F


class BaseNet(nn.Module):
    def __init__(self):
        """
        Base Network for all games
        https://github.com/YeWR/EfficientZero/blob/c533ebf5481be624d896c19f499ed4b2f7d7440d/core/model.py#L53
        """
        super(BaseNet, self).__init__()

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, momentum=0.1, downsample=None):
        """
        :param in_planes: number of input channels
        :param out_planes: number of output channels
        :param stride:
        :param downsample:
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes, momentum=momentum)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        skip = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            skip = self.downsample(x)

        out += skip
        return F.relu(out)
