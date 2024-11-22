import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from numpy import load

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class СustomAudioDataset(BaseDataset):
    """
    For custom dla_avss dataset
    """

    def __init__(
        self,
        part: str,
        dir: Optional[str],
        mouth_emb_dir: Optional[str] = None,
        use_mouth: bool = False,
        *args,
        **kwargs,
    ):
        if dir is None:
            data_dir = ROOT_PATH / "data" / "dla_dataset" / "audio"
        else:
            data_dir = ROOT_PATH / dir
        data_dir.mkdir(exist_ok=True, parents=True)
        self.mouth_emb_dir = mouth_emb_dir
        self._data_dir = data_dir
        self._part = part
        self._use_mouth = use_mouth
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
        mouth_dir = self.mouth_emb_dir
        if part == "train" or part == "val":
            s1_dir = split_dir / "s1"
            s2_dir = split_dir / "s2"

        for root, _, files in os.walk(mix_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if self._use_mouth:
                    mouth_path_s1 = os.path.join(mouth_dir, file.split("_")[0]) + ".npz"
                    mouth_path_s2 = os.path.join(mouth_dir, file.split("_")[1]).replace(
                        ".wav", ".npz"
                    )

                    paths = {
                        "path_mix": file_path,
                        "path_mouth_s1": mouth_path_s1,
                        "path_mouth_s2": mouth_path_s2,
                    }
                else:
                    paths = {
                        "path_mix": file_path,
                    }
                if part == "train" or part == "val":
                    s1_path = os.path.join(s1_dir, file)
                    s2_path = os.path.join(s2_dir, file)
                    paths.update(
                        {
                            "path_s1": s1_path,
                            "path_s2": s2_path,
                        }
                    )

                index.append(paths)

        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        mix_path = data_dict["path_mix"]
        mix_data_object = self.load_audio(mix_path)

        if self._part == "train" or self._part == "val":
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

        if self._use_mouth:
            mouth_data_s1 = load(data_dict["path_mouth_s1"])["data"]
            mouth_data_s2 = load(data_dict["path_mouth_s2"])["data"]
            instance_data["mouth_s1"] = mouth_data_s1
            instance_data["mouth_s2"] = mouth_data_s2
        return instance_data
