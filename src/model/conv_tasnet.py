import torch
from torch import nn


class BlockConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, sc_out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.PReLU(),
            gLN(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(dilation * (kernel_size - 1)) // 2, dilation=dilation),
            nn.PReLU(),
            gLN(out_channels),
        )
        self.next_block_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.skip_con_conv = nn.Conv1d(out_channels, sc_out_channels, 1, bias=True)

    def forward(self, x, skip_x):
        tmp = self.block(x)
        tmp = self.next_block_conv(tmp)
        new_skip_x = self.skip_con_conv(tmp)
        return x + tmp, new_skip_x + skip_x
    
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


class gLN(nn.Module):
    def __init__(self, nt, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(nt, 1))
        self.bias = nn.Parameter(torch.zeros(nt, 1))

        self.nt = nt
        self.eps = eps

    def forward(self, x):
        e = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - e)**2, (1, 2), keepdim=True)
        x = ((x - e) / torch.sqrt(var+self.eps)) * self.weight + self.bias
        return x


class ConvTasNetModel(nn.Module):
    """
    Conv-TasNet model
    """

    def __init__(self, N, L, B, Sc, H, P, X, R, num_speakers):
        """
        Args:
        
        N - Number of ﬁlters in autoencoder
        L - Length of the ﬁlters (in samples)
        B - Number of channels in bottleneck and the residual paths 1 x 1-conv blocks
        Sc - Number of channels in skip-connection paths 1 x 1-conv blocks
        H - Number of channels in convolutional blocks
        P - Kernel size in convolutional blocks
        X - Number of convolutional blocks in each repeat
        R - Number of repeats
        num_speakers - number of speakers in mix audio
        """
        super().__init__()

        self.num_speakers = num_speakers

        self.encoder = nn.Conv1d(1, N, L, stride=L // 2)
        
        self.start_separ = nn.Sequential(
            gLN(N),
            nn.Conv1d(N, B, 1)
        )

        self.tcn == nn.Sequential()
        for i in range(R):
            for j in range(X):
                self.tcn.append(BlockConv1D(B, H, P, j**2, Sc))
        self.end_separ = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(Sc, N * num_speakers, 1),
            nn.Sigmoid()
        )

        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=L//2)
        

    def forward(self, x):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        # encoder
        x = self.encoder(x)

        # separator
        emb = self.start_separ(x)

        _, separ_emb = self.tcn(emb)

        masks = self.end_separ(separ_emb)
        masks = torch.chunk(masks, chunks=self.num_speakers, dim=1)

        masked_emb = [x * masks[i] for i in range(self.num_speakers)]

        # decder
        result = [self.decoder(masked_emb[i]) for i in range(self.num_spks)]
        return result

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
