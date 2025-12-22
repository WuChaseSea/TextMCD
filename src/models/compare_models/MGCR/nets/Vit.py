# -*- coding: utf-8 -*-
import einops
import torch
import torch.nn as nn
import numpy as np
from timm.layers import DropPath
from torch.nn import Dropout, Conv2d
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Embeddings(nn.Module):
    # Construct the patch, position embeddings
    def __init__(self, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # [B, num_patches, hidden_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_patches, out_dim]
        x = self.act_layer(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, num_patches, 3*embed_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # [B, num_patches, 3, num_heads, per_HeadDim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, num_patches, per_HeadDim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, num_patches, per_HeadDim] [4, 8, 196, 8/16/32/64] easy to use tensor 

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, num_patches, num_patches]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)  # [B, num_heads, num_patches, per_HeadDim]
        x = x.transpose(1, 2)  # [B, num_patches, num_heads, per_HeadDim]
        x = x.reshape(B, N, C)  # [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, num_patches, embed_dim]
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_dim=dim, hidden_dim=self.mlp_hidden_dim, out_dim=dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


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


class VisionTransformer(nn.Module):
    def __init__(self, img_size, channel_num, patch_size, embed_dim, depth=1, num_heads=8,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super(VisionTransformer, self).__init__()
        self.embeddings = Embeddings(patch_size=patch_size, img_size=img_size, in_channels=channel_num)
        self.depth = depth
        self.dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Encoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(self.depth)])

        self.CTBN_i = ConvTransBN(in_channels=512, out_channels=512)
        self.CTBN_t = ConvTransBN(in_channels=64, out_channels=64)

        self.conv_down = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.dropout = nn.Dropout(0.1)


    def forward(self, img, text):
        H, W = img.shape[2], img.shape[3]
        img_res = img
        img = self.embeddings(img)

        img_text = img + self.CTBN_t(text)

        img_text = self.Encoder_blocks(img_text)

        img_text = img_text.transpose(1, 2)
        img_text = self.CTBN_i(img_text)
        img_text = img_text.transpose(1, 2)

        loss = []

        img_text = einops.rearrange(img_text, 'B (H W) C -> B C H W', H=H, W=W)

        loss.append(img_res)
        loss.append(img_text)
        out = self.conv_down(torch.cat([img_text, img_res], dim=1))  # [B, num_patches, embed_dim]

        out = self.dropout(out)
        return out, loss

