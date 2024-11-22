import torch
from torch import nn

from src.model import RTFSBlock
from src.model import gLN


class SSS(nn.Module):
    def __init__(self, Ca):
        """
        Args:
        
        
        """
        super().__init__()

        self.Ca = Ca

        self.generate_mask = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(Ca, Ca, 1),
            nn.ReLU()
        )
        

    def forward(self, ar, a0):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        mask = self.generate_mask(ar)

        mr = mask[:, 0:(self.Ca / 2 - 1)]
        mi = mask[:, self.Ca / 2:self.Ca]

        Er = a0[:, 0:(self.Ca / 2 - 1)]
        Ei = a0[:, self.Ca / 2:self.Ca]

        zr = mr * Er - mi * Ei
        zi = mr * Ei + mi * Er

        return torch.cat([zr, zi], dim=1)
        

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
    

class CAF(nn.Module):
    def __init__(self, Ca, Cv, h):
        """
        Args:
        
        
        """
        super().__init__()

        self.h = h
        
        self.p1 = nn.Sequential(
            nn.Conv2d(Ca, Ca, 1),
            nn.GroupNorm(num_groups=1, num_channels=Ca, eps=1e-6)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(Ca, Ca, 1),
            nn.GroupNorm(num_groups=1, num_channels=Ca, eps=1e-6),
            nn.ReLU()
        )

        self.f1 = nn.Sequential(
            nn.Conv1d(Cv, Ca * h, 1),
            gLN(Ca * h)
        )

        self.f2= nn.Sequential(
            nn.Conv1d(Cv, Ca, 1),
            gLN(Ca)
        )

        

    def forward(self, a, v):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        print("INIT SAF", a.shape, v.shape)
        a_val = self.p1(a)
        a_gate = self.p2(a)

        v_key = self.f2(v)

        vh = self.f1(v)
        vm = sum(torch.chunk(vh, chunks=self.h, dim=1)) / self.h

        v_attn = nn.functional.softmax(vm)
        print("CAF", v_attn.shape, a_val.shape)
        f1 = a_val * nn.functional.interpolate(v_attn, a_val.shape[2]).unsqueeze(-1)
        f2 = a_gate * nn.functional.interpolate(v_key, a_val.shape[2]).unsqueeze(-1)

        return f1 + f2
        

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


class Encoder(nn.Module):
    def __init__(self, in_chanels, out_chanels, n_fft, hop_length):
        """
        Args:
        
        
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.conv = nn.Conv2d(in_chanels, out_chanels, (3, 3), padding=1)
        self.window = torch.hann_window(n_fft)

    def forward(self, x):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        # print("emb lol", x.shape)
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device), return_complex=True, center=True)
        print("emb lol", x.shape)
        x = torch.stack([x.real, x.imag], 1).transpose(2, 3).contiguous() 
        print("emb lel", x.shape)
        return self.conv(x) 
        

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


class Decoder(nn.Module):
    def __init__(self, in_chanels, n_fft, hop_length):
        """
        Args:
        
        
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.conv = nn.ConvTranspose2d(in_chanels, 2, 3, padding=1)
        self.window = torch.hann_window(n_fft)

    def forward(self, x, audio_length=32000):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        tmp = x.shape[1]
        bs = x.shape[0]

        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x = self.conv(x)
        x = torch.complex(x[:, 0], x[:, 1]).transpose(1, 2).contiguous()

        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window.to(x.device), length=audio_length)

        return x.reshape(bs, tmp, audio_length) 
        

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


class RTFSNetModel(nn.Module):
    """
    Conv-TasNet model
    """

    def __init__(self, Ca=512, n_fft=256, hop_length=128, hidden_channels=64, kernel_size=3, caf_heads=4, vp_ch=50):
        """
        Args:
        
        
        """
        super().__init__()

        self.encoder = Encoder(in_chanels=2, out_chanels=Ca, n_fft=n_fft, hop_length=hop_length)

        self.vp = RTFSBlock(in_chanels=vp_ch, out_chanels=hidden_channels, kernel_size=kernel_size, upsampling_depth=4, n_heads=8, use_2d_conv=False, visual_part=True)  # 1d conv
        self.ap = RTFSBlock(in_chanels=Ca, out_chanels=hidden_channels, kernel_size=kernel_size, upsampling_depth=2, sru_num_layers=4, use_2d_conv=True)

        self.caf = CAF(Ca=Ca, Cv=vp_ch, h=caf_heads)

        self.decoder = Decoder(in_chanels=10, n_fft=n_fft, hop_length=hop_length)
        

    def forward(self, mix_data_object, mouth_s1, mouth_s2, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        paddings = torch.zeros(mix_data_object.shape[0], 704)
        mix_data_object = torch.cat([mix_data_object, paddings], dim=1)

        # mouth_s1
        a0 = self.encoder(mix_data_object)
        print("a0", a0.shape)
        a1 = self.ap(a0)
        print("a1", a1.shape)
        print("lol", mouth_s1.shape)
        v1 = self.vp(mouth_s1)

        aR = self.caf(a1, v1)

        # rtfs blocks
        for i in range(self.R):
            aR = self.ap(aR + a0)

        z = self.SSS(aR, a0)

        spk_1 = self.decoder(z)

        # mouth_s2
        a0 = self.encoder(mix_data_object)

        a1 = self.ap(a0)
        v1 = self.vp(mouth_s2)

        aR = self.caf(a1, v1)

        # rtfs blocks
        for i in range(self.R):
            aR = self.ap(aR + a0)

        z = self.SSS(aR, a0)

        spk_2 = self.decoder(z)
        return {"s1_pred_object": spk_1, "s2_pred_object": spk_2}
        

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
