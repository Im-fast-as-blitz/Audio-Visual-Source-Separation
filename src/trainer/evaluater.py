import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Evaluater(BaseTrainer):
    """
    Evaluater (Like Trainer but for evaluation) class

    Get metrics from model outputs and true separation.
    """

    def __init__(
        self,
        config,
        dataloaders,
        metrics=None,
    ):
        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.compute_metrics = config.inferencer.get("compute_metrics")

        # define dataloaders
        self.evaluation_dataloaders = dataloaders

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        return self._inference_part(self.evaluation_dataloaders)

    def process_batch(self, batch_idx, batch, metrics):
        if metrics is not None and self.compute_metrics:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))


        return batch

    def _inference_part(self, dataloader):
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
