import torch
from torchmetrics import SignalDistortionRatio
from src.metrics.base_metric import BaseMetric

class SDR(BaseMetric):
    def __init__(self, device:str, *args, **kwargs):
        """
        Computes SDR metric

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = SignalDistortionRatio().to(device)

    def __call__(self, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Takes predicted audio and target audio and returns SDR value.

        Args:
            preds (Tensor): predicted audio.
            target (Tensor): target audio.
        Returns:
            metric (float): calculated metric.
        """
        return self.metric(preds, target).mean().item()
    

class SDRi(BaseMetric):
    def __init__(self, device:str, *args, **kwargs):
        """
        Computes SDRi metric

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = SDR(device)

    def __call__(self, input: torch.Tensor, preds: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Takes input, predicted and target audio and returns SDRi value.

        Args:
            input (Tensor): input audio.
            preds (Tensor): predicted audio.
            target (Tensor): target audio.
        Returns:
            metric (float): calculated metric.
        """
        return self.metric(preds, target) - self.metric(input, target)