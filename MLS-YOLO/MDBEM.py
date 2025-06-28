import torch
import torch.nn as nn
import torch.nn.functional as F


class MDBEM(nn.Module):
    """
    Multi-Directional Boundary Enhancement Module (MDBEM)

    This is a plug-and-play module designed to enhance boundary information
    by computing directional differences across eight directions. It can be
    easily inserted into existing detection or segmentation networks, or
    used to replace specific layers where stronger boundary feature extraction
    is needed. Simply integrate MDBEM wherever additional boundary refinement
    is required to boost the model’s ability to capture fine-grained details.
    """
    def __init__(self, in_channels):
        super(MDBEM, self).__init__()

        # 1x1 convolution for initial feature transformation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Depthwise separable convolution for lightweight 3x3 processing
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, bias=False
        )

    def shift(self, x, direction):
        """
        Shift the feature map in the specified direction.
        Pads the empty region with zeros.
        """
        B, C, H, W = x.shape
        if direction == 'up':
            return F.pad(x[:, :, 1:, :], (0, 0, 0, 1), value=0)
        elif direction == 'down':
            return F.pad(x[:, :, :-1, :], (0, 0, 1, 0), value=0)
        elif direction == 'left':
            return F.pad(x[:, :, :, 1:], (0, 1, 0, 0), value=0)
        elif direction == 'right':
            return F.pad(x[:, :, :, :-1], (1, 0, 0, 0), value=0)
        elif direction == 'upleft':
            return F.pad(x[:, :, 1:, 1:], (0, 1, 0, 1), value=0)
        elif direction == 'upright':
            return F.pad(x[:, :, 1:, :-1], (1, 0, 0, 1), value=0)
        elif direction == 'downleft':
            return F.pad(x[:, :, :-1, 1:], (0, 1, 1, 0), value=0)
        elif direction == 'downright':
            return F.pad(x[:, :, :-1, :-1], (1, 0, 1, 0), value=0)

    def forward(self, x):
        # Initial 1x1 convolution
        feat = self.conv(x)

        # Compute directional differences for 8 directions
        directions = [
            'up', 'down', 'left', 'right',
            'upleft', 'upright', 'downleft', 'downright'
        ]

        direction_feats = []
        for d in directions:
            shifted = self.shift(feat, d)
            # Difference enhancement to highlight boundaries
            direction_feats.append(feat - shifted)

        # Aggregate all directional features
        boundary_feat = sum(direction_feats)

        # Refine boundary-enhanced features with depthwise separable convolution
        refined_feat = self.pointwise(self.depthwise(boundary_feat))

        # Residual connection
        return x + refined_feat
