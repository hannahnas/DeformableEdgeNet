import torch
import torch.nn as nn
from modules import ResNetBlockEnc, ResNetBlockDec

class SkipResNet(nn.Module):
    def __init__(self, deformable, use_edges, out_channels=1, act_fn=nn.ReLU):
        super().__init__()
        if use_edges:
            in_channels=5
        else:
            in_channels=4
        self.use_edges = use_edges

        self.enc_layer_0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            act_fn()
        )

        self.enc_layer_1 = nn.Sequential(
            ResNetBlockEnc(64, 64, act_fn, subsample=False),
            ResNetBlockEnc(64, 64, act_fn, subsample=False)
        )

        self.enc_layer_2 = nn.Sequential(
            ResNetBlockEnc(64, 128, act_fn, subsample=True, deformable=deformable),
            ResNetBlockEnc(128, 128, act_fn, subsample=False, deformable=deformable)
        )

        self.enc_layer_3 = nn.Sequential(
            ResNetBlockEnc(128, 256, act_fn, subsample=True, deformable=deformable),
            ResNetBlockEnc(256, 256, act_fn, subsample=False, deformable=deformable)
        )

        self.dec_layer_3 = nn.Sequential(
            ResNetBlockDec(256, 256, act_fn, subsample=False),
            ResNetBlockDec(256, 128, act_fn, subsample=True)
        )

        self.dec_layer_2 = nn.Sequential(
            ResNetBlockDec(256, 256, act_fn, subsample=False),
            ResNetBlockDec(256, 64, act_fn, subsample=True)
        )

        self.dec_layer_1 = nn.Sequential(
            ResNetBlockDec(128, 128, act_fn, subsample=False),
            ResNetBlockDec(128, 64, act_fn, subsample=True)
        )

        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self._init_params()

    def forward(self, batch):
        mask = batch['mask']
        img = batch['rgb']
        edge = batch['edges']
        depth = batch['depth']

        in_depth = (1 - mask) * depth
        
        if self.use_edges:
            in_feat = torch.cat([img, edge, in_depth], dim=1)
        else:
            in_feat = torch.cat([img, in_depth], dim=1)


        conv0 = self.enc_layer_0(in_feat) # B, 64, 128, 128
        conv1 = self.enc_layer_1(conv0) # B, 64, 128, 128
        conv2 = self.enc_layer_2(conv1) # B, 128, 64, 64
        conv3 = self.enc_layer_3(conv2) # B, 256, 32, 32
        
        upconv2 = self.dec_layer_3(conv3) # B, 128, 64, 64
        skipconv2 = torch.cat([conv2, upconv2], dim=1) # B, 256, 64, 64
        upconv1 = self.dec_layer_2(skipconv2) # B, 64, 128, 128
        skipconv1 = torch.cat([conv1, upconv1], dim=1) # B, 128, 128, 128
        upconv2 = self.dec_layer_1(skipconv1) # B, 64, 256, 256

        out = self.last_conv(upconv2) # B, 1, 256, 256

        return out

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    model = SkipResNet(deformable=False, use_edges=True, out_channels=1)

    x = torch.rand((4, 5, 256, 256))
    out = model(x)
    print(out.shape)




        

        