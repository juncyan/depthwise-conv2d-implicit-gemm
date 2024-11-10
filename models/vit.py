import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform
import numpy as np

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=3, img_size=224, patch_size=7, stride=4, embed_dim=768):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size // patch_size, img_size // patch_size
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
    def _init_weights(self, m):
        weight_attr = paddle.ParamAttr(initializer=1.0)
        bias_attr = paddle.framework.ParamAttr(initializer=0.0)
        if isinstance(m, nn.Linear):
            m.bias_attr = bias_attr
            m.weight_attr = weight_attr
        elif isinstance(m, nn.LayerNorm):
            m.bias_attr = bias_attr
            m.weight_attr = weight_attr
        elif isinstance(m, nn.Conv2D):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            
            m.weight_attr = paddle.normal(0, np.sqrt(2.0 / fan_out))
            # m.weight.data.normal_(0, np.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias_attr = paddle.framework.ParamAttr(initializer=0.0)