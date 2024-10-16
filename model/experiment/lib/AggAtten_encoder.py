import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_and_mask(input_resolution, window_size, device):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]], device=device), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2), stride=1)
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask


class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        self.is_extrapolation = is_extrapolation

        if not is_extrapolation:
            # The estimated training resolution is used for bilinear interpolation of the generated relative position bias.
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W
            self.trained_pool_H, self.trained_pool_W = input_resolution[0] // self.sr_ratio, input_resolution[
                1] // self.sr_ratio
            self.trained_pool_len = self.trained_pool_H * self.trained_pool_W

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative_bias_local:
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        B, N, C = x.shape
        pool_H, pool_W = H // self.sr_ratio, W // self.sr_ratio
        pool_len = pool_H * pool_W

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm = F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(padding_mask, float('-inf'))

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(self.act(self.sr(x_)), (pool_H, pool_W)).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        if self.is_extrapolation:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, N, pool_len)
        else:
            ##Use MLP to generate continuous relative positional bias for pooled features.
            pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                        relative_pos_index.view(-1)].view(-1, self.trained_len, self.trained_pool_len)

            # bilinear interpolation:
            pool_bias = pool_bias.reshape(-1, self.trained_len, self.trained_pool_H, self.trained_pool_W)
            pool_bias = F.interpolate(pool_bias, (pool_H, pool_W), mode='bilinear')
            pool_bias = pool_bias.reshape(-1, self.trained_len, pool_len).transpose(-1, -2).reshape(-1, pool_len,
                                                                                                    self.trained_H,
                                                                                                    self.trained_W)
            pool_bias = F.interpolate(pool_bias, (H, W), mode='bilinear').reshape(-1, pool_len, N).transpose(-1, -2)

        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(
            -2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, is_extrapolation=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AggregatedAttention(
            dim,
            input_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            is_extrapolation=is_extrapolation)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask):
        x = x + self.drop_path(
            self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class AAEcoder(nn.Module):

    def __init__(self, in_chans, embed_dims, input_resolution, out_resolution, num_heads, mlp_ratio, sr_ratio, drop_path, depth, pretrain_size=224, i=0,
                 window_size=3, patch_size=3, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm, qkv_bias=True, is_extrapolation=False):
        super().__init__()
        self.is_extrapolation = is_extrapolation
        self.i = i
        self.sr_ratio = sr_ratio
        self.patch_size = patch_size
        self.window_size = window_size
        self.pretrain_size = pretrain_size or input_resolution[0]
        self.patch_embed = OverlapPatchEmbed(patch_size=patch_size,
                                             stride=2,
                                             in_chans=in_chans,
                                             embed_dim=embed_dims)
        # self.block = Block(
        #         dim=embed_dims, input_resolution=out_resolution, window_size=window_size,
        #         num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer,
        #         sr_ratio=sr_ratio, is_extrapolation=is_extrapolation)
        self.block = nn.ModuleList([Block(
                dim=embed_dims, input_resolution=out_resolution, window_size=window_size,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer,
                sr_ratio=sr_ratio, is_extrapolation=is_extrapolation)
                for i in range(depth)])
        self.norm = norm_layer(embed_dims)

        if not self.is_extrapolation:
            relative_pos_index, relative_coords_table = get_relative_position_cpb(
                query_size=out_resolution,
                key_size=to_2tuple(out_resolution[0] // sr_ratio),
                pretrain_size=to_2tuple(self.pretrain_size // 2))

            self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)

        if self.is_extrapolation:
            relative_pos_index, relative_coords_table = get_relative_position_cpb(query_size=(H, W),
                                                                                  key_size=(
                                                                                      H // self.sr_ratio,
                                                                                      W // self.sr_ratio),
                                                                                  pretrain_size=to_2tuple(self.patch_size//2),
                                                                                  device=x.device)
        else:
            relative_pos_index = getattr(self, f"relative_pos_index{self.i + 1}")
            relative_coords_table = getattr(self, f"relative_coords_table{self.i + 1}")

        with torch.no_grad():
            local_seq_length, padding_mask = get_seqlen_and_mask((H, W), self.window_size, device=x.device)
            seq_length_scale = torch.log(local_seq_length + (H // self.sr_ratio) * (W // self.sr_ratio))

        for blk in self.block:
            x = blk(x, H, W, relative_pos_index, relative_coords_table, seq_length_scale, padding_mask)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x


if __name__ == '__main__':
    model = AAEcoder(in_chans=3, embed_dims=16, input_resolution=(512, 512), out_resolution=(256, 256),
                     num_heads=4, mlp_ratio=8, sr_ratio=8, drop_path=0.2, depth=2)
    device = torch.device('cuda:0')
    model = model.to(device=device)
    input_tensor = torch.randn(1, 3, 512, 512)
    input_tensor = input_tensor.to(device, dtype=torch.float)
    out = model(input_tensor)
    print(out.shape)