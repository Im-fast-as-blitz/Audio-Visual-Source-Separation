import torch_audiomentations
from torch import Tensor, nn


class ColoredNoise(nn.Module):
    """
    Add colored noise to the input audio.
    Instance transform.
    """

    def __init__(self, sample_rate, p, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(*args, sample_rate = sample_rate, p = p, **kwargs)

    def forward(self, input: Tensor):
        """
        Args:
            input (Tensor): input tensor.
        Returns:
            (Tensor): transformed tensor.
        """
        return self._aug(input.unsqueeze(1)).squeeze(1)
