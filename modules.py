import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.ops import DeformConv2d


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc=1, ks=3):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
 

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, mask):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        mask = (1 - mask)
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(mask)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, 1)
        self.norm_1 = SPADE(fmiddle, 1)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, 1)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2, mode='bilinear', batch_norm=True):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=1, padding=1, bias=False)
        self.batch_norm = batch_norm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.mode == 'bilinear':
            x = F.interpolate(x, scale_factor=self.scale,
                              mode=self.mode, align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode=self.mode)
        x = self.conv(x)

        if self.batch_norm:
            x = self.batchnorm2d(x)

        return x
        
        
class GatedDeformConvWithActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=nn.ReLU):
        super().__init__()
        self.batch_norm = batch_norm
        self.activation = activation()
        self.conv2d = BasicDeformConv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x
            
class GatedConv2dWithActivation(nn.Module):
    """
    Gated Convlution layer with activation
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True, activation=nn.ReLU):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

class ResNetBlockEnc(nn.Module):

    def __init__(self, c_in, c_out, act_fn=nn.ReLU, subsample=False, deformable=False):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        if deformable:
            self.net = nn.Sequential(
                # No bias needed as the Batch Norm handles it
                BasicDeformConv2d(c_in, c_out, kernel_size=3,
                               stride=1 if not subsample else 2),
                nn.BatchNorm2d(c_out),
                act_fn(),
                BasicDeformConv2d(c_out, c_out, kernel_size=3,
                               stride=1),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.net = nn.Sequential(
                # No bias needed as the Batch Norm handles it
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                          stride=1 if not subsample else 2, bias=False),
                nn.BatchNorm2d(c_out),
                act_fn(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(
            c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class ResNetBlockDec(nn.Module):

    def __init__(self, c_in, c_out, act_fn=nn.ReLU, subsample=False):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

            self.net = nn.Sequential(
                # No bias needed as the Batch Norm handles it
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                            stride=1, bias=False),
                nn.BatchNorm2d(c_out),
                act_fn(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.net = nn.Sequential(
                # No bias needed as the Batch Norm handles it
                UpConv(c_in, c_out, kernel_size=3),
                act_fn(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            )

        self.upsample = UpConv(c_in, c_out) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.upsample is not None:
            x = self.upsample(x)
        out = z + x
        out = self.act_fn(out)
        return out



class BasicDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, groups=1, offset_groups=1):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv2d_offset = nn.Conv2d(
            in_channels,
            offset_channels * offset_groups,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2d = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            bias=False
        )
    
    def forward(self, x):
        offset = self.conv2d_offset(x)
        return self.conv2d(x, offset)
