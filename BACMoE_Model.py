import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FPN_raw import FPN
###
# Paper: Boundary-Aware Collaborative Mixture-of-Experts
# Architecture for Fetal Brain Anatomical
# Segmentation
###

class IEAR(nn.Module):
    def __init__(self, channels=128, mid_channels=128, p=2.0, learn_p=True, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.in_proj = nn.Conv2d(channels, mid_channels, kernel_size=1)

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # soft interpolation scale
        self.a = nn.Parameter(torch.tensor(1.0))

        # learnable p (log-param for > 0 constraint)
        if learn_p:
            self.p_raw = nn.Parameter(torch.tensor(np.log(p), dtype=torch.float32))
        else:
            self.register_buffer('p_raw', torch.tensor(np.log(p), dtype=torch.float32))

        # optional SE-style channel attention
        if use_attention:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels // 4, mid_channels, kernel_size=1),
                nn.Sigmoid()
            )

    @property
    def p(self):
        return torch.exp(self.p_raw)

    def forward(self, x):
        x_proj = self.in_proj(x)

        x_clamped = torch.sigmoid(x).clamp(min=1e-4, max=1 - 1e-4)
        gate = (1 - x_clamped) ** self.p

        #enhanced = self.edge_enhance(x_proj)
        #fuzzy = (1 - gate) * x_proj + gate * self.a * enhanced  # soft interpolation
        fuzzy = gate * self.a * x_proj # enhanced
        if self.use_attention:
            fuzzy = fuzzy * self.attn(fuzzy)

        return self.edge_enhance(fuzzy) #+ x_proj


class FGEA(nn.Module):
    def __init__(self, channels=128, mid_channels=128, mode='none', low_thresh=0.1, high_thresh=0.3, learnable_mask=False):
        super().__init__()
        self.mode = mode
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.learnable_mask = learnable_mask
        self.in_proj = nn.Conv2d(channels, mid_channels, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # learnable parameter
        self.a = nn.Parameter(torch.tensor(1.0))

        # init
        if learnable_mask:
            self.mask_param = nn.Parameter(torch.ones(1, 1, 64, 64))  # 64×64 是预设大小

    def _make_frequency_mask(self, h, w, device):
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device) - h // 2,
            torch.arange(w, dtype=torch.float32, device=device) - w // 2,
            indexing='ij'
        )
        d = torch.sqrt(xx**2 + yy**2)
        d = d / d.max()

        if self.mode == 'low':
            mask = (d <= self.low_thresh).float()
        elif self.mode == 'high':
            mask = (d >= self.high_thresh).float()
        elif self.mode == 'band':
            mask = ((d >= self.low_thresh) & (d <= self.high_thresh)).float()
        else:  # 'none'
            mask = torch.ones_like(d)

        return mask[None, None, :, :]  # [1, 1, H, W]

    def forward(self, x):
        x_proj = self.in_proj(x)  # [B, C, H, W]
        B, C, H, W = x_proj.shape

        fft = torch.fft.fft2(x_proj, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        # build mask
        if self.learnable_mask:
            mask = F.interpolate(self.mask_param, size=(H, W), mode='bilinear', align_corners=False)
            mask = mask.expand(B, C, H, W).clamp(0, 1)
        else:
            mask = self._make_frequency_mask(H, W, x.device)
            mask = mask.expand(B, C, H, W)

        fft_filtered = fft_shifted * mask

        fft_unshifted = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        ifft = torch.fft.ifft2(fft_unshifted, dim=(-2, -1))
        x_reconstructed = ifft.real  # only take real part

        out = x_reconstructed# * self.a # + x_proj
        out = self.conv(out)
        return out


class GumbelRouter(nn.Module):
    def __init__(self, in_dim, num_experts, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.fc = nn.Linear(in_dim, num_experts)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.fc(x)
        
        # Normalize logits to prevent one-hot dominance
        logits = (logits - logits.mean(dim=-1, keepdim=True)) / (logits.std(dim=-1, keepdim=True) + 1e-6)

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        gate = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)

        # gate = straight_through_softmax((logits + gumbel_noise) / self.temperature, dim=-1)

        return gate, logits

def straight_through_softmax(logits, dim=-1):
    soft = F.softmax(logits, dim=dim)
    _, index = soft.max(dim=dim, keepdim=True)
    hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return (hard - soft).detach() + soft

def router_entropy_loss(gates):
    """
    gates: tensor, shape [B, num_experts]
    """
    p = gates.mean(dim=0)
    entropy = - (p * torch.log(p + 1e-9)).sum()
    return entropy

class BACMoE_Model(nn.Module):
    def __init__(self, k_fourier=5, k_fuzzy=5, k_evl=4, num_classes=1):
        super().__init__()
        self.encoder = FPN(hidden_dim=128, output_dim=128, pretrained=True)
        self.experts_fourier = nn.ModuleList([FGEA(mode='high') for _ in range(k_fourier)])
        self.experts_fuzzy = nn.ModuleList([IEAR() for _ in range(k_fuzzy)])
        self.router_fourier = GumbelRouter(in_dim=256, num_experts=k_fourier)
        self.router_fuzzy = GumbelRouter(in_dim=256, num_experts=k_fuzzy)
        self.conv_out = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        self.k = k_evl

    def forward(self, x, mode='train'):
        feats, rts = self.encoder(x)
        gate_fourier, logits_fourier = self.router_fourier(rts)
        gate_fuzzy, logits_fuzzy = self.router_fuzzy(rts)

        outputs_fourier = torch.stack([expert(feats) for expert in self.experts_fourier], dim=1)
        outputs_fuzzy = torch.stack([expert(feats) for expert in self.experts_fuzzy], dim=1)
        if mode == 'train':
            gate_fourier = gate_fourier.view(-1, gate_fourier.shape[1], 1, 1, 1)
            gate_fuzzy = gate_fuzzy.view(-1, gate_fuzzy.shape[1], 1, 1, 1)
            out_fourier = (outputs_fourier * gate_fourier).sum(dim=1)
            out_fuzzy = (outputs_fuzzy * gate_fuzzy).sum(dim=1)
            out = out_fourier + out_fuzzy

        elif mode == 'eval_topk':
            k = self.k
            out_list = []
            for i in range(x.size(0)):
                idx_f = gate_fourier[i].topk(k).indices
                idx_z = gate_fuzzy[i].topk(k).indices
                fused_f = (outputs_fourier[i, idx_f] * gate_fourier[i, idx_f].view(k, 1, 1, 1)).sum(dim=0)
                fused_z = (outputs_fuzzy[i, idx_z] * gate_fuzzy[i, idx_z].view(k, 1, 1, 1)).sum(dim=0)
                out_list.append(fused_f + fused_z)
            out = torch.stack(out_list)

        else:  # 'inference_top1'
            out = torch.stack([
                outputs_fourier[i, gate_fourier[i].argmax()] + outputs_fuzzy[i, gate_fuzzy[i].argmax()]
                for i in range(x.size(0))
            ])

        out = self.conv_out(out + feats)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        return out, (gate_fourier, gate_fuzzy)  # out, (logits_fourier, logits_fuzzy)



# x = torch.randn(1, 3, 416, 416)  # 假设为 [B, C, H, W]
# model = BACMoE_Model(num_classes=19)
# out, _ = model(x, mode='eval_topk')
# print(out.shape)
