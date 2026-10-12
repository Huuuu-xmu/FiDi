import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from torchvision.models.resnet import BasicBlock
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
        """ Content-aware feature reassembly (CARAFE).

        Wang et al., ICCV 2019: https://arxiv.org/abs/1905.02188.

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
        W = self.enc(W)  # b * (scale*k_up)^2 * h * w
        W = self.pix_shf(W)  # b * k_up^2 * h_ * w_
        W = F.softmax(W, dim=1)  # b * k_up^2 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * (k_up^2*c) * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * c * k_up^2 * h_ * w_

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

        # Use content-aware upsampling in the three coarse decoder stages.
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
        # ResNet-18 feature channels: 64, 64, 128, 256, 512.
        self.res2_conv = nn.Conv2d(64, 1, 1)
        self.res2_conv2 = nn.Conv2d(64, 1, 1)
        self.res3_conv = nn.Conv2d(128, 1, 1)
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


    def forward(self, x, res2, res3, use_adaup=True):
        # Restore the input resolution from the shallow feature map.
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
        self.gate_attention_256 = GroupedSpatialAttention(channels=256, groups=8)
        self.gate_attention_512 = GroupedSpatialAttention(channels=512, groups=8)

        self.gate_attention_112 = GroupedSpatialAttention(channels=112, groups=8)
        self.gate_attention_960 = GroupedSpatialAttention(channels=960, groups=8)

        self.use_cafe = True
        self.use_adaup = True
        self.use_indis = True

    def configure(self, *, use_cafe=True, use_adaup=True, use_indis=True):
        self.use_cafe = use_cafe
        self.use_adaup = use_adaup
        self.use_indis = use_indis


    def forward(self, *features):

        features = features[1:]  # Drop the input-resolution skip.
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        gate = None
        edge_feature = None
        if self.use_cafe:
            gate, edge_feature = self.shape_stream(
                features[4], features[3], features[2], use_adaup=self.use_adaup
            )

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None

            # Apply InDis to supported channel widths.
            if self.use_indis:
                if x.size(1) == 512:
                    x = self.gate_attention_512(x)
                if x.size(1) == 256:
                    x = self.gate_attention_256(x)
                if x.size(1) == 112:
                    x = self.gate_attention_112(x)
                if x.size(1) == 960:
                    x = self.gate_attention_960(x)

            x = decoder_block(x, skip, use_adaup=self.use_adaup)

        # Modulate decoder features with the CaFe boundary cue.
        if edge_feature is not None:
            x = x * (1.0 + edge_feature)

        return x, gate


class GroupedSpatialAttention(nn.Module):
    """Group-wise spatial gating following ULSAM equations 2-4.

    One grouped projection produces one attention map per channel group.
    Softmax is applied over spatial locations; the residual preserves shape.
    Method: Saini et al., WACV 2020, doi:10.1109/WACV45572.2020.9093341.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        if channels < 1 or groups < 1 or channels % groups:
            raise ValueError("channels must be positive and divisible by groups")
        self.channels = channels
        self.groups = groups
        self.channel_filter = nn.Conv2d(channels, channels, 1, groups=channels)
        self.channel_norm = nn.BatchNorm2d(channels, momentum=0.9)
        self.group_projection = nn.Conv2d(channels, groups, 1, groups=groups)
        self.group_norm = nn.BatchNorm2d(groups, momentum=0.9)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Concatenate the per-group tensors used by earlier FiDi checkpoints.
        if prefix + "subspaces.0.conv_dws.weight" in state_dict:
            names = {"channel_filter": "conv_dws", "channel_norm": "bn_dws",
                     "group_projection": "conv_point", "group_norm": "bn_point"}
            for destination, legacy in names.items():
                for key in getattr(self, destination).state_dict():
                    keys = [f"{prefix}subspaces.{group}.{legacy}.{key}"
                            for group in range(self.groups)]
                    if not all(key in state_dict for key in keys):
                        error_msgs.append(f"Incomplete legacy attention tensors for {prefix}")
                        continue
                    values = [state_dict.pop(key) for key in keys]
                    if values[0].ndim == 0:
                        if not all(torch.equal(values[0], value) for value in values):
                            error_msgs.append(f"Inconsistent legacy counters for {prefix}")
                        value = values[0]
                    else:
                        value = torch.cat(values, dim=0)
                    state_dict[prefix + destination + "." + key] = value
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("Expected an NCHW tensor with the configured channels")
        filtered = F.relu(self.channel_norm(self.channel_filter(features)))
        neighborhood = F.max_pool2d(filtered, kernel_size=3, stride=1, padding=1)
        scores = F.relu(self.group_norm(self.group_projection(neighborhood)))
        weights = scores.flatten(2).softmax(dim=-1).reshape_as(scores)
        batch, _, height, width = features.shape
        grouped = features.reshape(batch, self.groups, self.channels // self.groups, height, width)
        return (grouped * (1 + weights.unsqueeze(2))).reshape_as(features)
