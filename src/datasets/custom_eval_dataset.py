import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from numpy import load

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class СustomEvalAudioDataset(BaseDataset):
    """
    For custom dla_avss dataset
    """

    def __init__(
        self,
        dir_ideal: Optional[str],
        dir_predicted: Optional[str],
        target_sr=16000,
        *args,
        **kwargs,
    ):
        self._data_dir = ROOT_PATH / dir_ideal
        self._data_pred = ROOT_PATH / dir_predicted
        self.target_sr = target_sr
        self._index = self._get_or_load_index()

        # super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, ):
        index_path = self._data_pred / f"eval_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        dir_ideal_1 = self._data_dir / "s1"
        dir_ideal_2 = self._data_dir / "s2"
        dir_pred_1 = self._data_pred / "s1"
        dir_pred_2 = self._data_pred / "s2"

        for root, _, files in os.walk(dir_ideal_1):
            for file in files:
                file_path = os.path.join(root, file)
                ideal_2 = os.path.join(dir_ideal_2, file)
                pred_1 = os.path.join(dir_pred_1, "output_" + file)
                pred_2 = os.path.join(dir_pred_2, "output_" + file)
                
                index.append({"ideal1": file_path,
                              "ideal2": ideal_2,
                              "pred1": pred_1,
                              "pred2": pred_2})
        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        instance_data = {
            "s1_data_object" : self.load_audio(data_dict["ideal1"]),
            "s2_data_object" : self.load_audio(data_dict["ideal2"]),
            "s1_pred_object" : self.load_audio(data_dict["pred1"]),
            "s2_pred_object" : self.load_audio(data_dict["pred2"]),
        }

        return instance_data
