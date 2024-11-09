import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from src.metrics.base_metric import BaseMetric

class PESQ(BaseMetric):
    def __init__(self, device:str, fs: int = 16000, mode: str = 'wb', *args, **kwargs):
        """
        Computes PESQ metric

        Args:
            device (str): device for the metric calculation (and tensors).
            fs (int): sampling frequency.
            mode (str): wide-band or narrow-band.
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode).to(device)

    def __call__(self, mixed: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Takes mixed audio and target audio and returns PESQ value.

        Args:
            mixed (Tensor): mixed audio.
            target (Tensor): target audio.
        Returns:
            metric (float): calculated metric.
        """
        return self.metric(mixed, target).mean().item()