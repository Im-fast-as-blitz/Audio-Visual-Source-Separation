import logging
from typing import List, Optional
import json
import os
from pathlib import Path
from src.utils.io_utils import ROOT_PATH
from src.datasets.base_dataset import BaseDataset


logger = logging.getLogger(__name__)


class СustomAudioDataset(BaseDataset):
    """
    For custom dla_avss dataset 
    """
    def __init__(self, part: str, dir: Optional[str], *args, **kwargs):
        if dir is None:
            data_dir = ROOT_PATH / "data" / "dla_dataset" / "audio"
        else:
            data_dir = ROOT_PATH / dir
        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._part = part
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / "audio" / part

        mix_dir = split_dir / "mix"
        if part == "train":
            s1_dir = split_dir / "s1"
            s2_dir = split_dir / "s2"
        
        for root, dirs, files in os.walk(mix_dir):
            for file in files:
                file_path = os.path.join(root, file)

                if part == "train":
                    s1_path = os.path.join(s1_dir, file)
                    s2_path = os.path.join(s2_dir, file)

                    index.append(
                        {
                            "path_mix": file_path,
                            "path_s1": s1_path,
                            "path_s2": s2_path,
                        }
                    )
                else:
                    index.append({"path_mix": file_path})

        return index
    
    def __getitem__(self, ind):
        data_dict = self._index[ind]

        mix_path = data_dict["path_mix"]
        mix_data_object = self.load_audio(mix_path)

        if self._part == "train":
            s1_path = data_dict["path_s1"]
            s2_path = data_dict["path_s2"]
            s1_data_object = self.load_audio(s1_path)
            s2_data_object = self.load_audio(s2_path)
 

            instance_data = {
                            "mix_data_object": mix_data_object,
                            "s1_data_object": s1_data_object,
                            "s2_data_object": s2_data_object,
                            }
        else:
            instance_data = {"mix_data_object": mix_data_object}
        return instance_data
