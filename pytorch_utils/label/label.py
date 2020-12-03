import torch


def get_all_labels(ds: torch.utils.data.Dataset, label_index: int = 1, batch_size=64, num_workers=4) -> torch.Tensor:
    """
    Get all unique labels of a ds

    Args:
        ds: dataset
        label_index: tuple index of the labels. Defaults to 1 because most stereo-type return is x, y
        batch_size: 
        num_workers: number of workers for auxiliary dl

    """
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    label_cache = []
    
    for batch in dl:
        y = batch[label_index]
        label_cache.append(y.unique())
    
    label_cache = torch.cat(label_cache, 0)
    label_cache = label_cache.unique()

    return label_cache

