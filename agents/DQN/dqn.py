import math
from torch import nn
from torch.nn import functional as F
from agents.basenet import ResBlock, BaseNet


class DQNet(BaseNet):
    def __init__(self, obs_shape, action_dim, out_mlp_hidden_dim, num_blocks, res_out_channels: int, momentum=0.1):
        """
        Image-based
        Here I follows the model from paper EfficientZero
        Parameters
        ----------
        obs_shape: tuple or list
            shape of observations: [C, W, H]
        num_blocks: int
            number of res blocks
        out_mlp_hidden_dim: int
            channels of hidden states for the final hidden layer
        res_out_channels: bool
            True -> the number of out channels after the res net
        """
        super().__init__()

        # (3, 96, 96)
        self.conv1 = nn.Conv2d(obs_shape[0], res_out_channels // 2, kernel_size=3, stride=2, padding=1,
                               bias=False)
        # (out_channels//2, 48, 48)
        self.bn1 = nn.BatchNorm2d(res_out_channels // 2, momentum=momentum)
        self.block1 = ResBlock(res_out_channels // 2, res_out_channels // 2, momentum=momentum)
        self.ds_conv = nn.Conv2d(res_out_channels // 2, res_out_channels, kernel_size=3, stride=2, padding=1,
                                 bias=False)
        self.down_sample = ResBlock(res_out_channels // 2, res_out_channels, momentum=momentum, stride=2,
                                    downsample=self.ds_conv)
        # (out_channels, 24, 24)
        self.block2 = ResBlock(res_out_channels, res_out_channels, momentum=momentum)
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # (out_channels, 12, 12)
        self.bn2 = nn.BatchNorm2d(res_out_channels, momentum=momentum)
        self.block3 = ResBlock(res_out_channels, res_out_channels, momentum=momentum)
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # (out_channels, 6, 6)
        self.bn3 = nn.BatchNorm2d(res_out_channels, momentum=momentum)

        self.blocks = nn.ModuleList(
            [ResBlock(res_out_channels, res_out_channels, momentum=momentum) for _ in range(num_blocks)]
        )
        # observation_shape = (channel, height, width)
        self.dim_after_resnet = math.ceil(obs_shape[1] / 16) * math.ceil(obs_shape[2] / 16) * res_out_channels
        self.mlp = nn.Sequential(
            nn.Linear(self.dim_after_resnet, out_mlp_hidden_dim),
            nn.BatchNorm1d(out_mlp_hidden_dim, momentum=momentum),
            nn.ReLU(),
            nn.Linear(out_mlp_hidden_dim, action_dim),
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.down_sample(x)
        x = self.block2(x)
        x = F.relu(self.bn2(self.pooling1(x)))
        x = self.block3(x)
        x = F.relu(self.bn3(self.pooling2(x)))
        for block in self.blocks:
            x = block(x)

        x = self.mlp(x.view(-1, self.dim_after_resnet))
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean
