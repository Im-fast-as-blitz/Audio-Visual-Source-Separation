import torch
from torch import nn
from torchmetrics import ScaleInvariantSignalNoiseRatio
from src.metrics.base_metric import BaseMetric

class SI_SNR(nn.Module):
    def __init__(self, device:str = "auto", *args, **kwargs):
        """
        Computes SI-SNRi metric

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ScaleInvariantSignalNoiseRatio().to(device)

    def __call__(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Takes predicted audio and target audio and returns SI-SNRi value.

        Args:
            preds (Tensor): predicted audio.
            target (Tensor): target audio.
        Returns:
            metric (float): calculated metric.
        """
        return -self.metric(preds, target).mean()