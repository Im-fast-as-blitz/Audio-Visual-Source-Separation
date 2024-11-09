import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from src.metrics.base_metric import BaseMetric

class STOI(BaseMetric):
    def __init__(self, device:str, fs: int = 16000, *args, **kwargs):
        """
        Computes STOI metric

        Args:
            device (str): device for the metric calculation (and tensors).
            fs (int): sampling frequency.
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ShortTimeObjectiveIntelligibility(fs=fs).to(device)

    def __call__(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Takes predicted and target audio and returns STOI value.

        Args:
            preds (Tensor): predicted audio.
            target (Tensor): target audio.
        Returns:
            metric (float): calculated metric.
        """
        return self.metric(preds, target).mean().item()