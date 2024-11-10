import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from src.metrics.base_metric import BaseMetric
import itertools
import numpy as np


class STOI(BaseMetric):
    def __init__(self, device:str, num_speakers:int = 2, fs: int = 16000, *args, **kwargs):
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
        self.num_speakers = num_speakers

    def __call__(self, **kwargs):
        """
        Takes predicted and target audio and returns STOI value.

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
                    curr_metric += self.metric(kwargs[f"s{ind_pred+1}_pred_object"][val_ind], kwargs[f"s{ind_target+1}_data_object"][val_ind]).mean().item()
                metrics.append(curr_metric / self.num_speakers)
            batch_metrics.append(np.max(metrics))
        return np.mean(batch_metrics)
