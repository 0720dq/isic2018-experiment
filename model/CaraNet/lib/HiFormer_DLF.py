import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp, Block
import ssl
from model.CaraNet.lib.conv_layer import Conv

ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                               3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                  3)  # BNC -> BNH(C/H) -> BHN(C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class MultiScaleBlock(nn.Module):

    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches

        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[d] == dim[(d + 1) % num_branches] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[d]), act_layer(), nn.Linear(dim[d], dim[(d + 1) % num_branches])]
            self.projs.append(nn.Sequential(*tmp))

        self.fusion = nn.ModuleList()
        for d in range(num_branches):
            d_ = (d + 1) % num_branches
            nh = num_heads[d_]  # 3
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(
                    CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1], norm_layer=norm_layer,
                                        has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale,
                                                   drop=drop, attn_drop=attn_drop, drop_path=drop_path[-1],
                                                   norm_layer=norm_layer,
                                                   has_mlp=False))
                self.fusion.append(nn.Sequential(*tmp))

        self.revert_projs = nn.ModuleList()
        for d in range(num_branches):
            if dim[(d + 1) % num_branches] == dim[d] and False:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer(dim[(d + 1) % num_branches]), act_layer(),
                       nn.Linear(dim[(d + 1) % num_branches], dim[d])]
            self.revert_projs.append(nn.Sequential(*tmp))

    def forward(self, x):
        inp = x

        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(inp, self.projs)]
        # torch.Size([16, 1, 384])
        # torch.Size([16, 1, 96])

        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], inp[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, inp[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs

# DLF Module
class DLF(nn.Module):
    def __init__(self, num_patches=(64, 64), depth=[1, 1, 0], embed_dim=(32, 32), norm_layer=nn.LayerNorm, drop=0., drop_path=0., mlp_ratio=4):
        super().__init__()

        self.num_patches = num_patches
        num_branches = len(embed_dim)
        self.num_branches = num_branches

        # self.cls_token0 = nn.Parameter(torch.randn(1, 1, embed_dim[0]))  # nn.Parameter()定义可学习参数
        # self.cls_token1 = nn.Parameter(torch.randn(1, 1, embed_dim[1]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.line0 = nn.Linear(embed_dim[0], embed_dim[0])
        self.line1 = nn.Linear(embed_dim[1], embed_dim[1])

        self.blk = MultiScaleBlock(
            embed_dim, num_patches, depth, num_heads=(4, 4), mlp_ratio=(1., 1., 1.), qkv_bias=True,
            qk_scale=None, drop=0., attn_drop=0., drop_path=[0.0], norm_layer=norm_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm0 = norm_layer(embed_dim[0])
        mlp_hidden_dim0 = int(embed_dim[0] * mlp_ratio)
        self.mlp0 = Mlp(in_features=embed_dim[0], hidden_features=mlp_hidden_dim0, drop=drop)
        self.norm1 = norm_layer(embed_dim[1])
        mlp_hidden_dim1 = int(embed_dim[1] * mlp_ratio)
        self.mlp1 = Mlp(in_features=embed_dim[1], hidden_features=mlp_hidden_dim1, drop=drop)
        self.agg = aggregation(32)

    def forward(self, x1, x2):
        x1 = rearrange(x1, 'b c h w  -> b (h w) c')
        x2 = rearrange(x2, 'b c h w  -> b (h w) c')
        x = [x1, x2]
        cls_token0 = self.avgpool(x[0].transpose(1, 2))
        cls_token0 = rearrange(cls_token0, 'b c 1 -> b 1 c')
        cls_token1 = self.avgpool(x[1].transpose(1, 2))
        cls_token1 = rearrange(cls_token1, 'b c 1 -> b 1 c')
        # b, n, _ = x[0].shape  # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
        # cls_token0 = repeat(self.cls_token0, '() n d -> b n d', b=b)
        # cls_token1 = repeat(self.cls_token1, '() n d -> b n d', b=b)
        temp0 = torch.cat((cls_token0, x[0]), dim=1)  # 将cls_token拼接到patch token中去
        temp0 = self.line0(temp0)
        temp1 = torch.cat((cls_token1, x[1]), dim=1)
        temp1 = self.line1(temp1)
        xs = [temp0, temp1]

        xs = self.blk(xs)

        xs[0] = xs[0] + self.drop_path(self.mlp0(self.norm0(xs[0])))
        xs[1] = xs[1] + self.drop_path(self.mlp1(self.norm1(xs[1])))
        xs[0] = rearrange(xs[0][:, 1:, ...], 'b (h w) c -> b c h w', h=self.num_patches[0], w=self.num_patches[0])
        xs[1] = rearrange(xs[1][:, 1:, ...], 'b (h w) c -> b c h w', h=self.num_patches[1], w=self.num_patches[1])
        # print('xs[0].shape:', xs[0].shape)
        # print('xs[0].shape:', xs[1].shape)
        # out = self.agg(xs)

        return xs

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample = Conv(32, 32, 3, 1, padding=1)

        self.conv_concat = Conv(2*32, 2*32, 3, 1, padding=1)

        self.conv1 = Conv(2*32, 2*32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(2*32, 1, 1)

    def forward(self, x):
        x1_1 = x[1]
        x3_1 = self.conv_upsample(self.upsample(self.upsample(x[1]))) * x[0]

        x3_2 = torch.cat((x3_1, self.conv_upsample(self.upsample(self.upsample(x1_1)))), 1)
        x3_2 = self.conv_concat(x3_2)

        x = self.conv1(x3_2)
        x = self.conv2(x)

        return x


def main():
    model = DLF()

    # x1 = torch.randn(16, 3136, 96)
    # x2 = torch.randn(16, 196, 384)
    # x1 = torch.randn(16, 96, 56, 56)
    # x2 = torch.randn(16, 384, 14, 14)
    x1 = torch.randn(16, 32, 64, 64)
    x2 = torch.randn(16, 32, 64, 64)

    out = model(x1, x2)

    print(out[0].shape)
    print(out[1].shape)


if __name__ == '__main__':
    main()