# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/upernet_head.py

from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvModule(nn.Module):
    """Convolution module with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PPM(nn.Module):
    """Pyramid Pooling Module (PPM) used in PSPNet."""
    
    def __init__(self, pool_scales=(1, 2, 3, 6), in_channels=2048, channels=512):
        super().__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.ppm = nn.ModuleList()
        for pool_scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(in_channels, channels, kernel_size=1, padding=0)
                )
            )
        
        # Reduce channels after PPM
        self.bottleneck = ConvModule(
            in_channels + len(pool_scales) * channels,
            channels,
            kernel_size=3,
            padding=1
        )
        
    def forward(self, x):
        ppm_outs = [x]
        for ppm_module in self.ppm:
            ppm_out = ppm_module(x)
            ppm_out = F.interpolate(ppm_out, size=x.size()[2:], mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        output = self.bottleneck(ppm_outs)
        return output


class UPerHead(nn.Module):
    """Unified Perceptual Parsing head for segmentation.
    
    This head is implemented of `UPerNet <https://arxiv.org/abs/1807.10221>`_.
    """
    
    def __init__(
        self,
        input_shape: Dict[str, Tuple[int]],  # ShapeSpec: [channels, height, width, stride]
        in_channels: List[int] = [384, 768, 1024, 1536],  # Feature channels from backbone
        channels: int = 512,  # Intermediate channels
        pool_scales: Tuple[int] = (1, 2, 3, 6),
        num_classes: int = 150,
        ignore_value: int = 255,
        loss_weight: float = 1.0,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            in_channels: Input channels of multi-scale feature maps
            channels: Intermediate channels
            pool_scales: Pooling scales for PPM
            num_classes: Number of classes
            ignore_value: Category id to be ignored during training
            loss_weight: Loss weight
        """
        super().__init__()
        
        # Sort input shape by stride
        input_shape = sorted(input_shape.items(), key=lambda x: x[1][-1])
        self.in_features = [k for k, _ in input_shape]
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.ignore_value = ignore_value
        self.loss_weight = loss_weight
        
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channel in self.in_channels:
            l_conv = ConvModule(in_channel, channels, kernel_size=1, padding=0)
            fpn_conv = ConvModule(channels, channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
        # Self-attention refinement
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * channels,
            channels,
            kernel_size=3,
            padding=1
        )
        
        # PPM module
        self.ppm = PPM(pool_scales, channels, channels // 4)
        
        # Prediction head
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_out = self.ppm(x)
        return psp_out
    
    def _forward_feature(self, inputs):
        """Forward function for feature extraction."""
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=False)
            
        # Build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # Append the last feature map
        fpn_outs.append(laterals[-1])
        
        # Combine all FPN outputs
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=False)
            
        # Concat and bottleneck
        fuse_out = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fuse_out)
        
        return output
    
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.psp_forward([output])
        output = self.cls_seg(output)
        return output
    
    def predict(self, inputs, rescale_to=(512, 512)):
        """Prediction function with interpolation."""
        output = self.forward(inputs)
        output = F.interpolate(
            output,
            size=rescale_to,
            mode='bilinear',
            align_corners=False
        )
        return output
