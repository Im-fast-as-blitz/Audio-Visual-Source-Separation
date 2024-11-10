import torch
from torch import nn
from torchmetrics import ScaleInvariantSignalNoiseRatio
import itertools

class SI_SNR(nn.Module):
    def __init__(self, num_speakers:int = 2, device:str = "auto", *args, **kwargs):
        """
        Computes SI-SNRi metric

        Args:
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ScaleInvariantSignalNoiseRatio().to(device)
        self.num_speakers = num_speakers

    def __call__(self, **kwargs):
        """
        Takes predicted audio and target audio and returns SI-SNRi value.

        Args:
            kwargs
        Returns:
            metric (float): calculated metric.
        """
        batch_losses = []
        for val_ind in range(kwargs[f"s1_pred_object"].shape[0]):
            losses = []
            for perm in itertools.permutations(range(self.num_speakers)):
                curr_loss = 0
                for ind_target, ind_pred in enumerate(perm):
                    curr_loss += self.metric(kwargs[f"s{ind_pred+1}_pred_object"][val_ind], kwargs[f"s{ind_target+1}_data_object"][val_ind])
                losses.append(curr_loss / self.num_speakers)
            batch_losses.append(-torch.max(*losses))
        return {"loss": torch.mean(*batch_losses)}
