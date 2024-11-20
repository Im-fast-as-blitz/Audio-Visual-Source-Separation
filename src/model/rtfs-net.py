import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, ):
        """
        Args:
        
        
        """
        super().__init__()
        

    def forward(self, mix_data_object, mouth_emb, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        
        

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

        
        

    def forward(self, mix_data_object, mouth_emb, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        audio_enc = self.encoder(mix_data_object)

        a1 = self.ap(audio_enc)
        v1 = self.vp(mouth_emb)

        a2 = self.caf(a1, v1)



        return 
        

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
