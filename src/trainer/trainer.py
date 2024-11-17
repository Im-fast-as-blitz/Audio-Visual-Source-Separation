import pandas as pd

from random import shuffle
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # self._log_audio(batch)
            pass
        else:
            self._log_audio(batch)

    def _log_audio(self, batch, examples_to_log=8):
        result = {}
        examples_to_log = min(examples_to_log, batch['mix_data_object'].shape[0])

        tuples = list(zip(batch['mix_data_object'], batch['s1_data_object'], batch['s2_data_object'], 
                          batch['s1_pred_object'], batch['s2_pred_object']))
        shuffle(tuples)

        for idx, (mix, s1, s2, pred1, pred2) in enumerate(tuples[:examples_to_log]):
            result[idx] = {
                "mixed": self.writer.wandb.Audio(mix.squeeze(0).detach().cpu().numpy(), sample_rate=16000),
                "target_1": self.writer.wandb.Audio(s1.squeeze(0).detach().cpu().numpy(), sample_rate=16000),
                "target_2": self.writer.wandb.Audio(s2.squeeze(0).detach().cpu().numpy(), sample_rate=16000),
                "pred_1": self.writer.wandb.Audio(pred1.squeeze(0).detach().cpu().numpy(), sample_rate=16000),
                "pred_2": self.writer.wandb.Audio(pred2.squeeze(0).detach().cpu().numpy(), sample_rate=16000)
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(result, orient="index"))
                

