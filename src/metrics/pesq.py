import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from src.metrics.base_metric import BaseMetric
import itertools
import numpy as np


class PESQ(BaseMetric):
    def __init__(self, device:str, num_speakers: int = 2, fs: int = 16000, mode: str = 'wb', *args, **kwargs):
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
        self.num_speakers = num_speakers

    def __call__(self, **kwargs):
        """
        Takes pred audio and target audio and returns PESQ value.

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
