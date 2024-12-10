import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer.evaluater import Evaluater
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="only_metrics")
def main(config):

    # setup data_loader instances
    # batch_transforms should be put on device
    dataset = instantiate(config.datasets)

    # get metrics
    metrics = instantiate(config.metrics)
    print(dataset)
    inferencer = Evaluater(
        config=config,
        dataloaders=dataset,
        metrics=metrics,
    )

    logs = inferencer.run_inference()
    print(logs)


if __name__ == "__main__":
    main()
