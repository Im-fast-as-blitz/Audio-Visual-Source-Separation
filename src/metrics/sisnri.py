import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from src.metrics.base_metric import BaseMetric
import itertools
import numpy as np


class SI_SNR(BaseMetric):
    def __init__(self, device:str, *args, **kwargs):
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
        return self.metric(preds, target).mean().item()
    

class SI_SNRi(BaseMetric):
    def __init__(self, device:str, num_speakers:int = 2, *args, **kwargs):
        """
        Computes SI-SNRi metric

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = SI_SNR(device)
        self.num_speakers = num_speakers

    def __call__(self, **kwargs):
        """
        Takes input, predicted and target audio and returns SI-SNRi value.

        Args:
            kwargs
        Returns:
            metric (float): calculated metric.
        """
        batch_metrics = []
        for val_ind in range(kwargs[f"s1_pred_object"].shape[0]):
            metrics = []
            for perm in itertools.permutations(range(self.num_speakers)):
                curr_metric = 0
                for ind_target, ind_pred in enumerate(perm):
                    curr_metric += self.metric(kwargs[f"s{ind_pred+1}_pred_object"][val_ind], kwargs[f"s{ind_target+1}_data_object"][val_ind]) - self.metric(kwargs["mix_data_object"][val_ind], kwargs[f"s{ind_target+1}_data_object"][val_ind])
                metrics.append(curr_metric / self.num_speakers)
            batch_metrics.append(np.max(metrics))
        return np.mean(batch_metrics)
