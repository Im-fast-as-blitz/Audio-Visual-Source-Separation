import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from sru import SRU
import math
import numpy as np

from src.model import gLN


class ConvNorm(nn.Module):
    def __init__(self, in_chanel, out_chanels,
                kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                use_2d_conv=True,
                ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_chanel, out_chanels, kernel_size,
                      stride, padding, dilation, groups=groups) if use_2d_conv else nn.Conv1d(in_chanel, out_chanels, kernel_size,
                      stride, padding, dilation)
        # self.norm = gLN(out_chanels)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_chanels, eps=1e-6)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = nn.PReLU()(x)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        in_c, out_c,
        kernel_size = 5, dropout = 0.,
        padding=3,
        use_2d_conv=False,
        *args, **kwargs,
    ):
        super(FFN, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.conv1 = ConvNorm(in_c, out_c, 1, use_2d_conv=use_2d_conv)
        self.conv2 = ConvNorm(out_c, out_c, kernel_size, padding=padding, use_2d_conv=use_2d_conv)
        self.conv3 = ConvNorm(out_c, in_c, 1, use_2d_conv=use_2d_conv)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.conv1(x)
        print("conv2", x.shape)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        print(f"{skip.shape}, x.shape {x.shape}")
        x = self.drop(x) + skip
        return x


class DualPath(nn.Module):
    def __init__(self, in_c, out_c,
        dim, kernel_size = 8,
        stride = 1, num_layers = 1,
        use_2d_conv=True,
        *args, **kwargs,
    ):
        super(DualPath, self).__init__()
        self.out_c = out_c
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers


        self.unfold = nn.Unfold((self.kernel_size, 1), stride=(self.stride, 1))

        # ch = in_c * kernel_size
        ch = in_c
        self.ffn = FFN(ch, ch * 2, self.kernel_size, dropout=0.1, use_2d_conv=False)
        self.conv = nn.ConvTranspose1d(out_c * 2, in_c, self.kernel_size, stride=self.stride)

        self.rnn = SRU(
            input_size=ch * kernel_size,
            hidden_size=out_c,
            num_layers=self.num_layers,
            bidirectional=True,
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_c, eps=1e-6)

    def forward(self, x):
        print("init", x.shape)
        if self.dim == 4:
            x = x.transpose(-2, -1)

        if len(x.shape) == 4:
            bs, ch, t_in, f_in = x.shape
        else:
            bs, ch, t_in = x.shape
            f_in = t_in
        t_out = math.ceil((t_in - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        f_out = math.ceil((f_in - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        x = F.pad(x, (0, f_out - f_in, 0, t_out - t_in))

        print(x.shape)
        skip = x
        x = self.norm(x)

        x = x.permute(0, 3, 1, 2).reshape(bs * f_out, ch, t_out, 1)
        x = self.unfold(x)

        x = x.permute(2, 0, 1)
        x = self.rnn(x)[0]

        x = x.permute(1, 2, 0)
        x = self.ffn(x)
        x = self.conv(x)

        x = x.reshape(bs, f_out, ch, t_out)
        x = x.permute(0, 2, 3, 1)

        x = x + skip

        if self.dim == 4:
            x = x[:, :, :t_in, :f_in]
            x = x.transpose(-2, -1)
        else:
            x = x[:, :t_in, :f_in]

        return x


class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, in_channels, hidden_channels, N=4, use_2d_conv=True):
        super().__init__()

        self.N = N

        self.q_lay = nn.ModuleList([ConvNorm(in_channels, hidden_channels,
                                             kernel_size=1, use_2d_conv=use_2d_conv) for _ in range(N)])
        self.k_lay = nn.ModuleList([ConvNorm(in_channels, hidden_channels,
                                            kernel_size=1, use_2d_conv=use_2d_conv) for _ in range(N)])
        self.v_lay = nn.ModuleList([ConvNorm(in_channels, in_channels // N,
                                            kernel_size=1, use_2d_conv=use_2d_conv) for _ in range(N)])

        self.attn_proj = ConvNorm(in_channels, in_channels, kernel_size=1, use_2d_conv=use_2d_conv)

    def forward(self, x):
        b, c, t, f = x.shape

        print("mul had", x.shape)
        Q = torch.cat([lay(x) for lay in self.q_lay], dim=0).transpose(1, 2).flatten(start_dim=2)
        K = torch.cat([lay(x) for lay in self.k_lay], dim=0).transpose(1, 2).flatten(start_dim=2)
        V = torch.cat([lay(x) for lay in self.v_lay], dim=0).transpose(1, 2).flatten(start_dim=2)

        attn = F.softmax(Q @ K.transpose(1, 2) / (Q.shape[-1]**0.5), dim=2)

        V = attn @ V 
        V = V.reshape(b * self.N, t, c * f // self.N).transpose(1, 2)
        V = V.reshape([self.N, b, c // self.N, t, f]).transpose(0, 1)
        V = V.reshape([b, c, t, f])

        V = self.attn_proj(V)

        return V + x


class PosEncoder(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super().__init__()
        
        den = torch.exp(- torch.arange(0, emb_size, 2) * np.log(10000) / emb_size)
        pos = torch.arange(0, max_len).unsqueeze(1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        print("pos encoding", x.shape)
        out = x + self.pos_embedding[:, :x.size(1)]
        return out

class MultiHeadSelfAttention1D(nn.Module):
    def __init__(
        self,
        in_ch: int, out_ch: int,
        kernel_size: int,
        n_head: int = 8,
        dropout: int = 0.1,
        batch_first=True,
    ):
        super(MultiHeadSelfAttention1D, self).__init__()
        self.dropout = dropout
        self.batch_first = batch_first

        assert in_ch % n_head == 0, "In channels: {} must be divisible by the number of heads: {}".format(
            in_ch, n_head
        ) # TODO DELETE COMMENT

        self.norm1 = nn.LayerNorm(in_ch)
        self.norm2 = nn.LayerNorm(in_ch)

        self.positions = PosEncoder(in_ch)
        self.attention = nn.MultiheadAttention(in_ch, n_head, self.dropout, batch_first=self.batch_first)
        self.dropout = nn.Dropout(self.dropout)

        self.ffn = FFN(in_ch, out_ch, kernel_size, dropout=0.1, padding=2, use_2d_conv=False)
        

    def forward(self, x: torch.Tensor):
        res = x
        print("mha", x.shape)
        if self.batch_first:
            x = x.transpose(1, 2)

        x = self.norm1(x)
        x = self.positions(x)
        residual = x
        x = self.attention(x, x, x)[0]
        x = self.dropout(x) + residual
        x = self.norm2(x)

        if self.batch_first:
            x = x.transpose(2, 1)

        x = self.dropout(x) + res

        x = self.ffn(x)

        return x


class Reconstract(nn.Module):
    def __init__(
        self,
        in_chanel,
        kernel_size,
        padding=1,
        use_2d_conv=True,
    ):
        super().__init__()

        self.conv1 = ConvNorm(
            in_chanel=in_chanel,
            out_chanels=in_chanel,
            kernel_size=kernel_size,
            padding=padding,
            use_2d_conv=use_2d_conv,
        )
        self.conv2 = ConvNorm(
            in_chanel=in_chanel,
            out_chanels=in_chanel,
            kernel_size=kernel_size,
            padding=padding,
            use_2d_conv=use_2d_conv,
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chanel, in_chanel, kernel_size) if use_2d_conv else nn.Conv1d(in_chanel, in_chanel, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=1, num_channels=in_chanel, eps=1e-6),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        st_dim = -2 if len(x.shape) > 3 else -1

        out_dim = x.shape[st_dim:]

        if torch.prod(torch.tensor(out_dim)) > torch.prod(torch.tensor(skip.shape[st_dim:])):
            out = F.interpolate(self.conv2(skip), size=out_dim, mode="nearest")
            gate = F.interpolate(self.conv3(skip), size=out_dim, mode="nearest")
        else:
            g_interp = F.interpolate(skip, size=out_dim, mode="nearest")
            print("PAPA", g_interp.shape)
            out = self.conv2(g_interp)
            gate = self.conv3(g_interp)

        tmp = self.conv1(x)
        print(tmp.shape, gate.shape, out.shape)
        injection_sum = tmp * gate + out

        return injection_sum


class RTFSBlock(nn.Module):
    def __init__(
        self,
        in_chanels: int, out_chanels: int,
        kernel_size: int = 5, stride: int = 2,
        upsampling_depth: int = 2, n_heads: int = 4,
        use_2d_conv: bool = True,
        sru_num_layers: int = 4,
        visual_part: bool = False,
    ):
        super(RTFSBlock, self).__init__()
        self.in_chanels = in_chanels
        self.out_chanels = out_chanels
        self.upsampling_depth = upsampling_depth
        self.visual = visual_part
 
        self.pool = F.adaptive_avg_pool2d if not visual_part else F.adaptive_avg_pool1d
        self.skip = ConvNorm(
            in_chanels, in_chanels, kernel_size=1, groups=in_chanels, use_2d_conv=use_2d_conv
        )
        self.downsample1 = ConvNorm(
                                in_chanels, out_chanels, kernel_size=1, stride=1,  use_2d_conv=use_2d_conv
                            )
        
        self.downsample2 = ConvNorm(
                                out_chanels, out_chanels,
                                kernel_size=kernel_size,
                                padding=1,
                                stride=1,
                                use_2d_conv=use_2d_conv
                            )
        self.downsample3 = ConvNorm(
                                out_chanels, out_chanels,
                                kernel_size=kernel_size,
                                padding=1,
                                stride=stride, use_2d_conv=use_2d_conv
                            )
        
        self.dualpath1 = DualPath(out_chanels, 32, 4, 7, 1, sru_num_layers, use_2d_conv=use_2d_conv)  # magic constans :)
        self.dualpath2 = DualPath(out_chanels, 32, 3, 7, 1, sru_num_layers, use_2d_conv=use_2d_conv)
        self.attention = MultiHeadSelfAttention2D(out_chanels, 64, n_heads, use_2d_conv=use_2d_conv) if not visual_part else MultiHeadSelfAttention1D(out_chanels, out_chanels, kernel_size=5, n_head=n_heads)

        self.recon1_1 = Reconstract(out_chanels, kernel_size, use_2d_conv=use_2d_conv)
        self.recon1_2 = Reconstract(out_chanels, kernel_size, use_2d_conv=use_2d_conv)

        self.recon2 = Reconstract(out_chanels, kernel_size, use_2d_conv=use_2d_conv)
        self.residual_conv = ConvNorm(
                    out_chanels, in_chanels, 1, use_2d_conv=use_2d_conv
                )

    def forward(self, x: torch.Tensor):
        print("imput")
        skip = self.skip(x)

        print("before x", x.shape)
        x = self.downsample1(skip)
        print("x", x.shape)
        down1 = self.downsample2(x)
        print("down1", down1.shape)
        down2 = self.downsample3(down1)
        print("down2", down2.shape)

        if len(down2.shape) == 4:
            pool_size = (down2.shape[-2], down2.shape[-1])
        else:
            pool_size = down2.shape[-1]
        R2 = self.pool(down2, output_size=pool_size) + self.pool(down1, output_size=pool_size)


        if not self.visual:
            R = self.dualpath1(R2)
            R2 = self.dualpath2(R)
        A_hat = self.attention(R2)
        print(A_hat.shape)

        print("MAMA", down2.shape, A_hat.shape)
        A_1_1 = self.recon1_1(down1, A_hat)
        A_1_2 = self.recon1_2(down2, A_hat)
        print("A11", A_1_1.shape)
        print("A_1_2", A_1_2.shape)

        # fuse them into a single vector
        expanded = self.recon2(A_1_1, A_1_2) + down1

        tmp = self.residual_conv(expanded)
        print("expanded", tmp.shape)
        print("skip", skip.shape)

        out = tmp + skip

        return out
