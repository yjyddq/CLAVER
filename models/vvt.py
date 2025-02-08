from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential
import sys
sys.path.append("../")
from clip.model import LayerNorm, QuickGELU


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class SpatialAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath=0., ):
        super().__init__()
        '''Spatial'''
        self.attn = nn.MultiheadAttention(d_model, n_head, )
        self.ln_1 = LayerNorm(d_model)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        '''Spatial'''
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, droppath=0., T=0, ):
        super().__init__()
        self.T = T
        self.H = n_head
        '''Temporal'''
        self.temporal_attn = nn.MultiheadAttention(d_model, n_head, )
        self.temporal_ln_1 = LayerNorm(d_model)

        self.temporal_drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()
        self.temporal_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.temporal_ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def temporal_attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.temporal_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        '''Temporal'''
        l, bt, d = x.size()
        b = bt // self.T

        x = x.permute(1, 0, 2)  # [BT,L,D]
        x = x.reshape(b, self.T * l, d)  # [BT,L,D] -> [B,TL,D]
        x = x.permute(1, 0, 2)  # [B,TL,D] -> [TL,B,D]
        x = x + self.temporal_drop_path(self.temporal_attention(self.temporal_ln_1(x)))
        x = x + self.temporal_drop_path(self.temporal_mlp(self.temporal_ln_2(x)))
        x = x.permute(1, 0, 2)  # [TL,B,D] -> [B,TL,D]
        x = x.reshape(bt, l, d)  # [B,TL,D] -> [BT,L,D]
        x = x.permute(1, 0, 2)  # [BT,L,D] -> [L,BT,D]
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None,
                 use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)]
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(
            *[SpatialAttentionBlock(width, heads, attn_mask, droppath[i], ) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        if not self.use_checkpoint:
            return self.resblocks(x)
        else:
            return checkpoint_sequential(self.resblocks, 3, x)

class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, droppath=None,
                 use_checkpoint=False, T=8):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if droppath is None:
            droppath = [0.0 for i in range(layers)]
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(
            *[TemporalAttentionBlock(width, heads, attn_mask, droppath[i], T) for i in range(layers // 3)])
            
    def forward(self, x: torch.Tensor):
        if not self.use_checkpoint:
            return self.resblocks(x)
        else:
            return checkpoint_sequential(self.resblocks, 3, x)


class VideoVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 attn_mask: torch.Tensor = None, attn_mask_t: torch.Tensor = None, droppath=None, T=8,
                 use_checkpoint=False, ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.T = T
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.temporal_embedding = nn.Parameter(scale * torch.randn((1, T, width)))
        trunc_normal_(self.temporal_embedding, std=0.02)
        self.ln_pre = LayerNorm(width)

        ## Attention Blocks
        self.transformer = SpatialTransformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, T=T,
                                        attn_mask=attn_mask)
        self.transformer_t = TemporalTransformer(width, layers, heads, droppath=droppath, use_checkpoint=use_checkpoint, T=T,
                                          attn_mask=attn_mask_t)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [BT, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [BT, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [BT, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [BT, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # [NLD] -> [LND]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [LND] -> [NLD]

        x = x.reshape(x.shape[0] // self.T, self.T, x.shape[1], -1)  # shape = [BT//T, T, grid ** 2 + 1, width]

        x = x + self.temporal_embedding.unsqueeze(2).to(x.dtype)
        x = x.reshape(x.shape[0] * self.T, x.shape[2], -1)  # shape = [BT, grid ** 2 + 1, width]
        x = x.permute(1, 0, 2)  # [NLD] -> [LND]
        x = self.transformer_t(x)
        x = x.permute(1, 0, 2)  # [LND] -> [NLD]

        cls_x = self.ln_post(x[:, 0, :])  # shape = [BT, width]

        if self.proj is not None:
            cls_x = cls_x @ self.proj

        return cls_x, x[:, 1:, :] # [BT, width] [BT, grid ** 2, width]


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt

    def kronecker(A, B):
        AB = torch.einsum("ab,cd->acbd", A, B)
        AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
        return AB

    def upper_triangle_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask

    def build_temporal_mask(T, L, fill_value='-inf'):
        '''
        attn: [B, H, TL, TL]
        L: patch nums or node nums
        T: time slots
        '''
        mask = torch.ones((L, L)) - torch.eye(L)
        diag = torch.eye(T)
        attn_mask = kronecker(diag, mask)
        if fill_value == '-inf':
            attn_mask[attn_mask == 1] = - torch.inf
        elif fill_value == 'zero':
            attn_mask[attn_mask == 1] = 0.
        return attn_mask

    def build_causal_temporal_mask(T, L, fill_value='-inf'):
        '''
        attn: [B, H, TL, TL]
        L: patch nums
        T: time slots
        '''
        mask = torch.ones((L, L)) - torch.eye(L)
        diag = torch.eye(T)
        attn_mask = kronecker(diag, mask)

        causal_mask = torch.ones((L, L))
        up_triu = upper_triangle_mask(T)
        attn_causal_mask = kronecker(up_triu, causal_mask)
        attn_mask += attn_causal_mask
        if fill_value == '-inf':
            attn_mask[attn_mask == 1] = - torch.inf
        elif fill_value == 'zero':
            attn_mask[attn_mask == 1] = 0.
        return attn_mask

    torch.manual_seed(10)
    l, t, b, d = 4, 3, 1, 1
    attn_mask_kmt = build_temporal_mask(t, l)
    attn_mask_kmct = build_causal_temporal_mask(t, l)
    KMTA = TemporalAttentionBlock(d_model=1, n_head=1, attn_mask=attn_mask_kmt, droppath=0., T=t)
    KMCTA = TemporalAttentionBlock(d_model=1, n_head=1, attn_mask=attn_mask_kmct, droppath=0., T=t)

    x = torch.randn(size=(b, t, l, d), dtype=torch.float).view(b, t * l, d)
    x = x.permute(1, 0, 2)
    print('input shape:', x.shape)

    '''rank of KMT'''
    y, attn = KMTA.temporal_attn(x, x, x, need_weights=True, attn_mask=KMTA.attn_mask)
    print(attn>1e-12)
    sns.heatmap(attn.squeeze(0).detach().cpu().numpy(), cmap='Blues')
    plt.show()

    '''rank of KMCT'''
    y, attn = KMCTA.temporal_attn(x, x, x, need_weights=True, attn_mask=KMCTA.attn_mask)
    print(attn > 1e-12)
    sns.heatmap(attn.squeeze(0).detach().cpu().numpy(), cmap='Blues')
    plt.show()