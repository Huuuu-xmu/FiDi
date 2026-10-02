import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from torchvision.models.resnet import BasicBlock
import matplotlib.pyplot as plt

def  draw_tensor(tensor, sample, dim, cmap=None):
    if len(tensor.size()) == 3:
        pic = tensor[sample, :, :].cpu().detach().numpy()
    else:
        pic = tensor[sample,dim,:,:].cpu().detach().numpy()

    if cmap != None:
        plt.imshow(pic, cmap=cmap)
    else:
        plt.imshow(pic)
    plt.show()


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=3, k_enc=3):
        """ The unofficial implementation of the CARAFE module.

        The details are in "https://arxiv.org/abs/1905.02188".

        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X






class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
        use_adaptive_module=True,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

        # The manuscript schematic shows three AdaUp paths. Restrict the
        # content-aware module to the three coarse decoder stages; finer stages
        # use nearest-neighbor interpolation to keep batch-16 memory bounded.
        self.adaptive_upsampler = (
            CARAFE(in_channels, c_mid=min(in_channels, 64))
            if use_adaptive_module else None
        )

    def forward(self, x, skip=None, use_adaup=True):
        x = self.adaptive_upsampler(x) if use_adaup and self.adaptive_upsampler is not None else F.interpolate(
            x, scale_factor=2, mode="nearest"
        )

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class ShapeStream(nn.Module):
    def __init__(self):
        super().__init__()
        # for resnet18
        # mobilenetv3: 3, 16, 24, 40, 112, 960
        # resnet18: 3, 64, 64, 128, 256, 512
        self.res2_conv = nn.Conv2d(64, 1, 1)
        self.res2_conv2 = nn.Conv2d(64, 1, 1)
        self.res3_conv = nn.Conv2d(128, 1, 1)
        # self.res4_conv = nn.Conv2d(256, 1, 1)
        self.res1 = BasicBlock(64, 64, 1)
        self.res2 = BasicBlock(64, 64, 1)
        self.res3 = BasicBlock(128, 128, 1)
        self.res1_pre = nn.Conv2d(64, 32, 1)
        self.res2_pre = nn.Conv2d(64, 32, 1)
        self.res3_pre = nn.Conv2d(128, 64, 1)
        self.gate3 = GatedConv(64, 64)
        self.gate2 = GatedConv(32, 32)

        self.gate = nn.Conv2d(32, 1, 1, bias=False)
        self.fuse = nn.Conv2d(2, 1, 1, bias=False)
        self.adaup_x = CARAFE(1, c_mid=16)
        self.adaup_res2 = CARAFE(1, c_mid=16)
        self.adaup_res3 = CARAFE(64, c_mid=64)

        # for mobilenetv3::::
        # mobilenetv3: 3, 16, 24, 40, 112, 960
        # self.res2_conv = nn.Conv2d(16, 1, 1)
        # self.res2_conv2 = nn.Conv2d(24, 1, 1)
        # self.res3_conv = nn.Conv2d(40, 1, 1)
        # # self.res4_conv = nn.Conv2d(256, 1, 1)
        # self.res1 = BasicBlock(64, 64, 1)
        # self.res2 = BasicBlock(64, 64, 1)
        # self.res3 = BasicBlock(128, 128, 1)
        # self.res1_pre = nn.Conv2d(40, 24, 1)
        # self.res2_pre = nn.Conv2d(24, 16, 1)
        # self.res3_pre = nn.Conv2d(40, 24, 1)
        # self.gate3 = GatedConv(24, 24)
        # self.gate2 = GatedConv(16, 16)
        #
        # self.gate = nn.Conv2d(16, 1, 1, bias=False)
        # self.fuse = nn.Conv2d(2, 1, 1, bias=False)



    def forward(self, x, res2, res3, use_adaup=True):
        # The shallow feature is at half input resolution. This retains the
        # LSLE 448x448 output while allowing the 512x512 Genova input size.
        size = (x.size(-2) * 2, x.size(-1) * 2)

        res3 = self.res3_pre(res3)
        res2 = self.res2_conv2(res2)
        x = self.res2_conv(x)
        if use_adaup:
            res3 = self.adaup_res3(res3)
            res2 = self.adaup_res2(res2)
            x = self.adaup_x(x)
        res3 = F.interpolate(res3, size, mode='bilinear', align_corners=True)
        res2 = F.interpolate(res2, size, mode='bilinear', align_corners=True)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        gate3 = self.gate3(res3, res2)
        gate2 = self.gate2(self.res2_pre(gate3), x)

        gate = torch.sigmoid(self.gate(gate2))
        feat = torch.sigmoid(self.fuse(torch.cat((gate, gate), dim=1)))
        return gate, feat


class GatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1, bias=False)
        self.attention = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, feat, gate):
        attention = self.attention(torch.cat((feat, gate), dim=1))
        out = F.conv2d(feat * (attention + 1), self.weight)
        return out


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_adaptive_module=i < 3, **kwargs)
            for i, (in_ch, skip_ch, out_ch) in enumerate(zip(in_channels, skip_channels, out_channels))
        ]
        self.blocks = nn.ModuleList(blocks)

        self.shape_stream = ShapeStream()

        # self.gate_attention_32 = ULSAM(nin=64,nout=64,h=224,w=224,num_splits=4)
        # self.gate_attention_64 = ULSAM(nin=64, nout=64, h=112, w=112, num_splits=4)
        # self.gate_attention_128 = ULSAM(nin=128, nout=128, h=56, w=56, num_splits=4)
        self.gate_attention_256 = ULSAM(nin=256, nout=256, h=28, w=28, num_splits=8)
        self.gate_attention_512 = ULSAM(nin=512, nout=512, h=14, w=14, num_splits=8)

        self.gate_attention_112 = ULSAM(nin=112, nout=112, h=28, w=28, num_splits=8)
        self.gate_attention_960 = ULSAM(nin=960, nout=960, h=14, w=14, num_splits=8)

        self.use_cafe = True
        self.use_adaup = True
        self.use_indis = True

    def configure(self, *, use_cafe=True, use_adaup=True, use_indis=True):
        self.use_cafe = use_cafe
        self.use_adaup = use_adaup
        self.use_indis = use_indis


    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution   mobilenetv3: 3,16,24,40,112,960   resnet18: 3,64,64,128,256,512
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        # for TMI method
        gate = None
        edge_feature = None
        if self.use_cafe:
            gate, edge_feature = self.shape_stream(
                features[4], features[3], features[2], use_adaup=self.use_adaup
            )
        # draw_tensor(gate, 0, 0)

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None

            # for latent space split
            if self.use_indis:
                if x.size(1) == 512:
                    x = self.gate_attention_512(x)
                if x.size(1) == 256:
                    x = self.gate_attention_256(x)
                if x.size(1) == 112:
                    x = self.gate_attention_112(x)
                if x.size(1) == 960:
                    x = self.gate_attention_960(x)
            # if x.size(1) == 128:
            #     x = self.gate_attention_128(x)

            x = decoder_block(x, skip, use_adaup=self.use_adaup)

        # Fuse the high-resolution CaFe cue into the decoder output, as shown
        # in the manuscript's technical route. This residual modulation is a
        # source-derived implementation choice; the final fusion code is lost.
        if edge_feature is not None:
            x = x * (1.0 + edge_feature)

        return x, gate





class SubSpace(nn.Module):
    """
    Subspace class.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int) -> None:
        super(SubSpace, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.bn_dws = nn.BatchNorm2d(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dws(x)
        out = self.bn_dws(out)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.bn_point(out)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out


class ULSAM(nn.Module):
    """
    Grouped Attention Block having multiple (num_splits) Subspaces.

    ...

    Attributes
    ----------
    nin : int
        number of input feature volume.

    nout : int
        number of output feature maps

    h : int
        height of a input feature map

    w : int
        width of a input feature map

    num_splits : int
        number of subspaces

    Methods
    -------
    __init__(nin)
        initialize method.
    forward(x)
        forward pass.

    """

    def __init__(self, nin: int, nout: int, h: int, w: int, num_splits: int) -> None:
        super(ULSAM, self).__init__()

        assert nin % num_splits == 0

        self.nin = nin
        self.nout = nout
        self.h = h
        self.w = w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_size = int(self.nin / self.num_splits)

        # split at batch dimension
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)

        # out = 1. + torch.tanh(out)

        return out
