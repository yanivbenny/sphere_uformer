from typing import Tuple, Any, Dict, Optional

from torch.utils.data import DataLoader

from .stanford2d3d import Stanford2D3D


def get_dataloaders(
        dataset_name: str,
        dataset_root_dir: Optional[str],
        dataset_kwargs: Dict[str, Any],
        augmentation_kwargs: Dict[str, Any],
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:

    if dataset_name.lower() == "stanford2d3d":
        dataset_train = Stanford2D3D(
            root_dir=dataset_root_dir,
            list_file="./data/splits_2d3d/stanford2d3d_train.txt",
            dataset_kwargs=dataset_kwargs,
            augmentation_kwargs=augmentation_kwargs,
            is_training=True,
        )
        dataset_val = Stanford2D3D(
            root_dir=dataset_root_dir,
            list_file="./data/splits_2d3d/stanford2d3d_val.txt",
            dataset_kwargs=dataset_kwargs,
            augmentation_kwargs=augmentation_kwargs,
            is_training=False,
        )
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")


    # Configure dataloaders
    loader_train = DataLoader(
        dataset_train,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
    )

    loader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,
    )

    return loader_train, loader_val
