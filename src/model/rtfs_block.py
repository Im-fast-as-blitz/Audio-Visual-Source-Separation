import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
from conv_tasnet import gLN
from sru import SRU
import math


class ConvNorm(nn.Module):
    def __init__(self, in_chanels, out_chanels,
                kernel, stride=1, padding=0, dilation=1,
                use_2d_conv=True,
                *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(in_chanels, out_chanels, kernel,
                      stride, padding, dilation) if use_2d_conv else nn.Conv1d(in_chanels, out_chanels, kernel,
                      stride, padding, dilation)
    
    def forward(self, x):
        x = self.conv(x)
        x = gLN(x)
        x = nn.PReLU(x)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        in_c, out_c,
        kernel_size = 5, dropout = 0.,
        use_2d_conv=True,
        *args, **kwargs,
    ):
        super(FFN, self).__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.conv1 = ConvNorm(in_c, out_c, 1, use_2d_conv=use_2d_conv)
        self.conv2 = ConvNorm(out_c, out_c, kernel_size, use_2d_conv=use_2d_conv)
        self.conv3 = ConvNorm(out_c, in_c, 1, use_2d_conv=use_2d_conv)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
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

        ch = in_c * kernel_size
        self.ffn = FFN(ch, ch * 2, self.kernel_size, dropout=0.1, use_2d_conv=use_2d_conv)
        self.conv = nn.ConvTranspose1d(self.out_c * 2, self.in_chanel, self.kernel_size, stride=self.stride)

        self.rnn = SRU(
            input_size=ch,
            hidden_size=self.hid_chan,
            num_layers=self.num_layers,
            bidirectional=True,
        )

    def forward(self, x):
        if self.dim == 4:
            x = x.transpose(-2, -1)

        bs, ch, t_in, f_in = x.shape
        t_out = math.ceil((t_in - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        f_out = math.ceil((f_in - self.kernel_size) / self.stride) * self.stride + self.kernel_size
        x = F.pad(x, (0, f_out - f_in, 0, t_out - t_in))

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

        Q = torch.cat([lay(x) for lay in self.q_lay], dim=0).transpose(1, 2).flatten(start_dim=2)
        K = torch.cat([lay(x) for lay in self.k_lay], dim=0).transpose(1, 2).flatten(start_dim=2)
        V = torch.cat([lay(x) for lay in self.v_lay], dim=0).transpose(1, 2).flatten(start_dim=2)

        attn = F.softmax(Q @ K.transpose(1, 2) / (Q.shape[-1]**0.5), dim=2)

        V = attn @ V 
        V = V.reshape(b * self.N, t, c * f // self.N).transpose(1, 2)
        V = V.view([self.N, b, c // self.N, t, f]).transpose(0, 1)
        V = V.view([b, c, t, f])

        V = self.attn_proj(V)

        return V + x


class Reconstract(nn.Module):
    def __init__(
        self,
        in_chanel,
        kernel_size,
        use_2d_conv=True,
    ):
        super(Reconstract, self).__init__()

        self.conv1 = ConvNorm(
            in_chanel=in_chanel,
            out_chan=in_chanel,
            kernel_size=kernel_size,
            use_2d_conv=use_2d_conv,
        )
        self.conv2 = ConvNorm(
            in_chanel=self.in_chanel,
            out_chan=self.in_chanel,
            kernel_size=self.kernel_size,
            use_2d_conv=use_2d_conv,
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chanel, in_chanel, kernel_size) if use_2d_conv else nn.Conv1d(in_chanel, in_chanel, kernel_size),
            gLN(),
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
            out = self.conv2(g_interp)
            gate = self.conv3(g_interp)

        injection_sum = self.conv1(x) * gate + out

        return injection_sum


class RTFSBlock(nn.Module):
    def __init__(
        self,
        in_chanels: int, out_chanels: int,
        kernel_size: int = 5, n_heads:int = 4,
        upsampling_depth: int = 2, use_2d_conv: bool = True
    ):
        super(RTFSBlock, self).__init__()
        self.in_chanels = in_chanels
        self.out_chanels = out_chanels
        self.upsampling_depth = upsampling_depth
 
        self.pool = F.adaptive_avg_pool2d

        self.skip = ConvNorm(
            in_chanels, in_chanels, kernel_size=1, use_2d_conv=use_2d_conv
        )
        self.downsample1 = ConvNorm(
                                in_chanels, out_chanels, kernel_size=1, use_2d_conv=use_2d_conv
                            )
        self.downsample2 = ConvNorm(
                                out_chanels, out_chanels,
                                kernel_size=kernel_size,
                                stride=0, use_2d_conv=use_2d_conv
                            )
        self.downsample3 = ConvNorm(
                                out_chanels, out_chanels,
                                kernel_size=kernel_size,
                                stride=2, use_2d_conv=use_2d_conv
                            )

        self.dualpath1 = DualPath(out_chanels, 32, 4, 8, 1, use_2d_conv=use_2d_conv)  # magic constans :)
        self.dualpath2 = DualPath(out_chanels, 32, 3, 8, 1, use_2d_conv=use_2d_conv)
        self.attention = MultiHeadSelfAttention2D(32, 64, n_heads, use_2d_conv=use_2d_conv)

        self.recon1_1 = Reconstract(out_chanels, kernel_size, use_2d_conv=use_2d_conv)
        self.recon1_2 = Reconstract(out_chanels, kernel_size, use_2d_conv=use_2d_conv)

        self.recon2 = Reconstract(out_chanels, kernel_size, use_2d_conv=use_2d_conv)
        self.residual_conv = ConvNorm(
                    out_chanels, in_chanels, 1, use_2d_conv=use_2d_conv
                )

    def forward(self, x: torch.Tensor):
        skip = self.skip(x)

        x = self.downsample1(skip)
        down1 = self.downsample2(x)
        down2 = self.downsample3(down1)

        assert len(down2.shape) == 4
        pool_size = (down2.shape[-2], down2.shape[-1])
        A = self.pool(down2, output_size=pool_size) + self.pool(down1, output_size=pool_size) 
        R = self.dualpath1(A)
        R2 = self.dualpath2(R)
        A_hat = self.attention(R2)

        A_1_1 = self.recon1_1(down1, A_hat)
        A_1_2 = self.recon1_2(down2, A_hat)

        # fuse them into a single vector
        expanded = self.recon2[-1](A_1_1, A_1_2) + down1

        out = self.residual_conv(expanded) + skip

        return out

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

