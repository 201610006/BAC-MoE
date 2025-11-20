import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet18

class FPN(nn.Module):

    def __init__(self, hidden_dim=128, output_dim=128, pretrained=False):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.back_bone = ResNet18(pretrained=pretrained)

        # Top layer
        self.toplayer = nn.Conv2d(2048, hidden_dim, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, hidden_dim, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.gn1 = nn.GroupNorm(output_dim, output_dim)
        self.gn2 = nn.GroupNorm(hidden_dim, hidden_dim)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, x):
        # Bottom-up using backbone
        low_level_features = self.back_bone(x)

        c1 = low_level_features[0]
        c2 = low_level_features[1]
        c3 = low_level_features[2]
        c4 = low_level_features[3]
        c5 = low_level_features[4]

        # Top-down
        p5 = self.toplayer(c5)

        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        # return self._upsample(self.conv3(s2 + s3 + s4 + s5), 4*h, 4*w)
        return s2 + s3 + s4 + s5, p5
    
