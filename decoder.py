import torch
import torch.nn as nn
from modules import SPADEResnetBlock, UpConv, ResNetBlockDec

class ResNetDecoder(nn.Module):
    def __init__(self, out_channels=16, act_fn=nn.ReLU):
        super().__init__()
        self._create_network(out_channels, act_fn=act_fn)
        self._init_params()

    def _create_network(self, out_channels, act_fn):

        self.layer_3 = nn.Sequential(
            ResNetBlockDec(256, 256, act_fn, subsample=False),
            ResNetBlockDec(256, 128, act_fn, subsample=True)
        )

        self.layer_2 = nn.Sequential(
            ResNetBlockDec(128, 128, act_fn, subsample=False),
            ResNetBlockDec(128, 64, act_fn, subsample=True)
        )

        self.layer_1 = nn.Sequential(
            ResNetBlockDec(64, 64, act_fn, subsample=False),
            ResNetBlockDec(64, 32, act_fn, subsample=True)
        )

        self.layer_0 = nn.Sequential(
            ResNetBlockDec(32, 32, act_fn, subsample=False),
            ResNetBlockDec(32, 32, act_fn, subsample=True)
        )

        self.last_conv = nn.Conv2d(32, out_channels, 3, padding=1)

        # self.layer_0 = nn.Sequential(
        #     nn.Conv2d(64, in_channels, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64),
        #     act_fn()
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
        x = self.layer_3(x)
        x = self.layer_2(x)
        x = self.layer_1(x)
        x = self.layer_0(x)
        x = self.last_conv(x)
        return x


# class SPADEDecoder(nn.Module):
#     def __init__(self, in_channels, out_channels, act_fn=nn.ReLU):
#         super().__init__()

#         self.layer_3 = nn.Sequential(
#             SPADEResnetBlock(256, 256, act_fn),
#             UpConv(256, 128),
#             act_fn()
#         )

#         self.layer_2 = nn.Sequential(
#             SPADEResnetBlock(128, 128, act_fn),
#             UpConv(128, 64),
#             act_fn()
#         )

#         self.layer_1 = nn.Sequential(
#             SPADEResnetBlock(64, 64, act_fn),
#             nn.Conv2d(64, 64),
#             nn.BatchNorm2d(64),
#             act_fn()
#         )

#     def forward(self, x):
#         x = self.layer_3(x)
#         x = self.layer_2(x)
#         x = self.layer_1(x)
#         return x



if __name__ == '__main__':
    decoder = ResNetDecoder(out_channels=1)

    x = torch.rand((4, 256, 16, 16))
    out = decoder(x)
    print(out.shape)
