import torch
from torch import nn

from conv_tasnet import gLN
from rtfs_block import RTFSBlock


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
            gLN(Ca)
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(Ca, Ca, 1),
            gLN(Ca),
            nn.ReLU()
        )

        self.f1 = nn.Sequential(
            nn.Conv1d(Cv, Ca * h, 1),
            gLN(Ca)
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
        a_val = self.p1(a)
        a_gate = self.p2(a)

        v_key = self.f2(v)

        vh = self.f1(v)
        vm = torch.mean(torch.chunk(vh, chunks=self.h, dim=1))

        v_attn = nn.functional.softmax(vm)
        f1 = a_val * nn.functional.interpolate(v_attn, (v_attn.shape[0], v_attn.shape[1], a_val.shape[2]))
        f2 = a_gate * nn.functional.interpolate(v_key, (v_key.shape[0], v_key.shape[1], a_val.shape[2]))

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
    def __init__(self, in_chanels, out_chanels):
        """
        Args:
        
        
        """
        super().__init__()
        
        self.conv = nn.Conv2d(in_chanels, out_chanels, (3, 3))

    def forward(self, x):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        print("emb lol", x.shape)
        x = torch.stft(x, n_fft=self.win, hop_length=self.hop_length, window=self.window.to(x.device), return_complex=True)
        x = torch.stack([x.real, x.imag], 1)   # .transpose(2, 3).contiguous() 
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


class RTFSNetModel(nn.Module):
    """
    Conv-TasNet model
    """

    def __init__(self, ):
        """
        Args:
        
        
        """
        super().__init__()

        self.encoder = Encoder(in_chanels=, out_chanels=)

        self.ap = RTFSBlock(in_chanels=, out_chanels=, kernel_size = 5, stride = 2, upsampling_depth = 2, use_2d_conv = True)
        self.vp = RTFSBlock(in_chanels=, out_chanels=, kernel_size = 5, stride = 2, upsampling_depth = 2, use_2d_conv = False)   # 1d conv

        self.decoder = nn.ConvTranspose2d(, 2, 3)

        self.caf = CAF(Ca=)

        

    def forward(self, mix_data_object, mouth_emb, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        a0 = self.encoder(mix_data_object)

        a1 = self.ap(a0)
        v1 = self.vp(mouth_emb)

        aR = self.caf(a1, v1)

        # rtfs blocks
        for i in range(self.R):
            aR = self.ap(aR + a0)

        z = self.SSS(aR, a0)

        return self.decoder(z)
        

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
