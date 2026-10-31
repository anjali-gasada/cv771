import numpy as np
import torch
import torch.nn as nn
import math

from .blocks import (PatchEmbed, window_partition, window_unpartition,
                    DropPath, MLP, trunc_normal_)


################################################################################
# You will need to fill in the missing code in this file
################################################################################


################################################################################
# Part I.1: A simple convolutional neural network
################################################################################
class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
################################################################################
# Part I.2: Design your own CNN - IMPROVED ARCHITECTURES
################################################################################


class SimpleNetBN(nn.Module):
    """
    SimpleNet with Batch Normalization.

    IMPROVEMENT: BatchNorm after each Conv, before ReLU
    - Faster convergence (can use higher learning rates)
    - Acts as regularization (reduces need for dropout)
    - Stabilizes training (reduces internal covariate shift)

    Expected improvement: +2-5% accuracy over baseline SimpleNet
    """
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNetBN, self).__init__()

        self.features = nn.Sequential(
            # conv1 block: 7x7 conv with BatchNorm
            conv_op(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2 bottleneck with BatchNorm
            conv_op(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv3 bottleneck with BatchNorm
            conv_op(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """
    Basic Residual Block with skip connection.

    Implements: output = F(x) + shortcut(x)
    where F = Conv-BN-ReLU-Conv-BN

    Skip connections allow gradients to flow directly backward,
    enabling training of deeper networks.
    """
    def __init__(self, in_channels, out_channels, stride=1, drop_path_rate=0.0):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Stochastic Depth (DropPath) for regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # Shortcut: identity if dims match, else 1x1 projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out = self.relu(out + identity)
        return out


class ImprovedCNN(nn.Module):
    """
    Improved CNN combining best practices from the lecture:

    1. Batch Normalization: faster training, implicit regularization
    2. Skip Connections (ResNet-style): better gradient flow, deeper networks
    3. Stochastic Depth (DropPath): regularization, implicit ensemble
    4. He Initialization: proper weight scaling for ReLU
    5. Dropout before classifier: prevents overfitting

    Architecture: Stem → 4 stages of residual blocks → GAP → FC
    Total depth: ~18 conv layers

    Expected improvement: +5-10% accuracy over baseline SimpleNet
    """
    def __init__(self, num_classes=100, drop_path_rate=0.1):
        super(ImprovedCNN, self).__init__()

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )  # 128 → 32

        # Stochastic depth with linear decay schedule
        total_blocks = 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Residual stages with increasing channels
        self.layer1 = self._make_layer(64, 64, 2, stride=1, drop_rates=dpr[0:2])
        self.layer2 = self._make_layer(64, 128, 2, stride=2, drop_rates=dpr[2:4])   # 32 → 16
        self.layer3 = self._make_layer(128, 256, 2, stride=2, drop_rates=dpr[4:6])  # 16 → 8
        self.layer4 = self._make_layer(256, 512, 2, stride=2, drop_rates=dpr[6:8])  # 8 → 4

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, drop_rates):
        layers = [ResidualBlock(in_ch, out_ch, stride, drop_rates[0])]
        for i in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1, drop_rates[i]))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class InceptionBlock(nn.Module):
    """
    Inception-style block with parallel branches for multi-scale features.

    Four parallel branches:
    1. 1x1 conv: point-wise features
    2. 1x1 → 3x3 conv: local patterns
    3. 1x1 → 5x5 conv: larger patterns
    4. 3x3 pool → 1x1 conv: pooled features

    Outputs are concatenated along channel dimension.
    """
    def __init__(self, in_ch, ch1x1, ch3x3_red, ch3x3, ch5x5_red, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, ch1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 → 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, ch3x3_red, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch3x3_red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_red, ch3x3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 → 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, ch5x5_red, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch5x5_red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_red, ch5x5, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # Branch 4: pool → 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                         self.branch3(x), self.branch4(x)], dim=1)


class InceptionNet(nn.Module):
    """
    Inception-style network with parallel branches for multi-scale features.

    Architecture:
    - Stem: 7x7 conv + pool → 64 channels
    - Inception blocks with increasing channels
    - Global average pooling + dropout + FC

    Benefits: captures features at multiple scales simultaneously
    """
    def __init__(self, num_classes=100):
        super(InceptionNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # InceptionBlock(in, 1x1, 3x3_red, 3x3, 5x5_red, 5x5, pool_proj)
        self.inception1 = InceptionBlock(64, 64, 64, 96, 16, 32, 32)   # out: 224
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception2 = InceptionBlock(224, 128, 96, 128, 32, 64, 64)  # out: 384
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3 = InceptionBlock(384, 160, 128, 192, 48, 96, 64)  # out: 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.pool2(x)
        x = self.inception3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ============================================================================
# CHANGE THIS TO SELECT YOUR MODEL!
# Options:
#   - SimpleNet: Original baseline (no BatchNorm, no skip connections)
#   - SimpleNetBN: Baseline + BatchNorm (easiest improvement)
#   - ImprovedCNN: ResNet-style with BatchNorm + skip connections + DropPath
#   - InceptionNet: Multi-scale parallel branches
# ============================================================================
default_cnn_model = ImprovedCNN

################################################################################
# Part I.3: Understanding self-attention and Transformer block
################################################################################
class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        """
        Args:
            dim (int): Number of input channels. We assume Q, K, V will be of
                same dimension as the input.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # linear projection for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # linear projection at the end
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # input size (B, H, W, C)
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(
                B, H * W, 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        #################################################################################
        # Fill in the code here
        #################################################################################
        return x

class TransformerBlock(nn.Module):
    """Transformer blocks with support of local window self-attention"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window_size=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            window_size (int): Window size for window attention blocks.
                If it equals 0, then global attention is used.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path>0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer
        )

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        ########################################################################
        # Fill in the code here
        ########################################################################
        # The implementation shall support local self-attention
        # (also known as window attention)
        return x


#################################################################################
# Part I.4: Design and train a vision Transformer
#################################################################################
class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(0, 2),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for local attention blocks.
            window_block_indexes (list): Indexes for blocks using local attention.
                Local window attention allows more efficient computation, and
                can be coupled with standard global attention. E.g., [0, 2]
                indicates the first and the third blocks will use local window
                attention, while other block use standard attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()
        assert (num_heads > 0) and (embed_dim % num_heads == 0) 

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # patch embedding layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ########################################################################
        # Fill in the code here
        ########################################################################
        # The implementation shall define some Transformer blocks


        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ########################################################################
        # Fill in the code here
        ########################################################################

        return x

# change this to your model!
default_vit_model = SimpleViT



