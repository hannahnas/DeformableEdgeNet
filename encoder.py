import torch.nn as nn
import torch
from modules import ResNetBlockEnc


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, deformable=False, act_fn=nn.ReLU):
        super().__init__()
        self._create_network(in_channels, deformable=deformable, act_fn=act_fn)
        self._init_params()

    def _create_network(self, in_channels, deformable, act_fn):

        self.layer_0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer_1 = nn.Sequential(
            ResNetBlockEnc(64, 64, act_fn, subsample=False),
            ResNetBlockEnc(64, 64, act_fn, subsample=False)
        )

        self.layer_2 = nn.Sequential(
            ResNetBlockEnc(64, 128, act_fn, subsample=True, deformable=deformable),
            ResNetBlockEnc(128, 128, act_fn, subsample=False, deformable=deformable)
        )

        self.layer_3 = nn.Sequential(
            ResNetBlockEnc(128, 256, act_fn, subsample=True, deformable=deformable),
            ResNetBlockEnc(256, 256, act_fn, subsample=False, deformable=deformable)
        )

        # self.layer_4 = nn.Sequential(
        #     ResNetBlockEnc(256, 512, act_fn, subsample=True,
        #                    deformable=deformable),
        #     ResNetBlockEnc(512, 512, act_fn, subsample=False,
        #                    deformable=deformable)
        # )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # x = self.layer_4(x)

        return x



if __name__ == '__main__':
    deform_encoder = ResNetEncoder(in_channels=3, deformable=True)
    encoder = ResNetEncoder(in_channels=3, deformable=False)

    x = torch.rand((4, 3, 256, 256))
    out = deform_encoder(x)
    print(out.shape)

