import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # result_batch["spectrogram"] = pad_sequence([item["spectrogram"].permute(2, 1, 0) for item in dataset_items], batch_first=True).squeeze(-1).permute(0, 2, 1)
    # result_batch["spectrogram_length"] = torch.tensor([item["spectrogram"].shape[2] for item in dataset_items])

    result_batch["mix_data_object"] = pad_sequence(
        [item["mix_data_object"].squeeze(0) for item in dataset_items], batch_first=True
    )
    if "s1_data_object" in dataset_items[0]:
        result_batch["s1_data_object"] = pad_sequence(
            [item["s1_data_object"].squeeze(0) for item in dataset_items],
            batch_first=True,
        )
        result_batch["s2_data_object"] = pad_sequence(
            [item["s2_data_object"].squeeze(0) for item in dataset_items],
            batch_first=True,
        )
    else:
        result_batch["s1_data_object"] = torch.tensor([])
        result_batch["s2_data_object"] = torch.tensor([])
    if "mouth_s1" in dataset_items[0]:
        result_batch["mouth_s1"] = pad_sequence(
            [torch.Tensor(item["mouth_s1"]).squeeze(0) for item in dataset_items], batch_first=True
        )
        result_batch["mouth_s2"] = pad_sequence(
            [torch.Tensor(item["mouth_s2"]).squeeze(0) for item in dataset_items], batch_first=True
        )
    result_batch["id"] = [item["id"] for item in dataset_items]
    return result_batch
