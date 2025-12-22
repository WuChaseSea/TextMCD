import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Conv2d
from torch.nn.modules.utils import _pair


class SpectralGCN(nn.Module):
    def __init__(self, in_dim, N, ratio=4):
        super(SpectralGCN, self).__init__()
        inter_dim = in_dim // ratio

        self.phi = nn.Linear(in_dim, inter_dim, bias=False)
        self.bn_phi = nn.LayerNorm(inter_dim)

        self.theta = nn.Linear(in_dim, inter_dim, bias=False)
        self.bn_theta = nn.LayerNorm(inter_dim)

        self.conv_adj = nn.Conv1d(in_channels=N, out_channels=N, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(N)

        self.conv_wg = nn.Conv1d(N * 2, N * 2, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(N * 2)

        self.proj = nn.Linear(N * 2, in_dim, bias=False)

    def forward(self, x):
        # Feature Projection
        x_phi = self.bn_phi(self.phi(x))
        x_theta = self.bn_theta(self.theta(x))

        # Affinity Matrix
        z_idt = torch.bmm(x_phi, x_theta.transpose(1, 2))

        # Graph Interaction (Laplace + conv)
        z = z_idt.transpose(1, 2).contiguous()
        z = self.conv_adj(z)
        z = self.bn_adj(z)
        z = z.transpose(1, 2).contiguous()
        z = z + z_idt

        # Feature Fusion
        feat_z = torch.cat([z, z_idt], dim=2)
        feat_z = feat_z.transpose(1, 2)
        feat_z = self.bn_wg(self.conv_wg(feat_z))
        feat_z = feat_z.transpose(1, 2)

        # Project back to C
        out = self.proj(feat_z)

        # Residual + Activation
        return F.relu(x + out)


class Embeddings(nn.Module):
    def __init__(self, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

# Semantic Graph-Conditioned Module
class SGCM(nn.Module):
    def __init__(self,  img_size, channel_num, patch_size, embed_dim, num_heads=8):
        super(SGCM, self).__init__()
        self.embeddings = Embeddings(patch_size=patch_size, img_size=img_size, in_channels=channel_num)

        self.linear_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.attn_text = nn.MultiheadAttention(embed_dim, num_heads//2, dropout=0.1)
        self.attn_img = nn.MultiheadAttention(embed_dim, num_heads//2, dropout=0.1)

        self.cross_gcn = SpectralGCN(in_dim=embed_dim, N=img_size * img_size)

    def forward(self, img, text_feat):
        # b,c,h,w = img.shape
        H, W = img.shape[2], img.shape[3]
        img_feat = self.embeddings(img) # b, n, c

        # Graph-Conditioned
        cat_VL = torch.cat([img_feat, text_feat], dim=2)
        down_VL = self.linear_proj(cat_VL)
        key_graph_nodes = self.cross_gcn(down_VL)

        # Vision-Language Reconstruction
        img_feat =  self.attn_img(img_feat.permute(1, 0, 2), key_graph_nodes.permute(1, 0, 2), key_graph_nodes.permute(1, 0, 2))[0].permute(1, 0, 2)
        text_feat =  self.attn_text(text_feat.permute(1, 0, 2), key_graph_nodes.permute(1, 0, 2), key_graph_nodes.permute(1, 0, 2))[0].permute(1, 0, 2)

        img_feat = einops.rearrange(img_feat, 'B (H W) C -> B C H W', H=H, W=W)

        return text_feat, img_feat