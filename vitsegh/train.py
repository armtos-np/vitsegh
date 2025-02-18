####################### For Google Colab ########################
import sys
import os

# Add the parent directory of 'vitsegh' to the Python path
module_path = os.path.abspath("/content/vitsegh/")
if module_path not in sys.path:
    sys.path.append(module_path)

####################### For Google Colab ########################

import gc
import warnings  # Added here due to a SyntaxWarning raised by a pre-trained model

# See discussion here: https://discuss.pytorch.org/t/strange-warning-when-working-with-pretrained-models/162327/4
warnings.filterwarnings("ignore", category=SyntaxWarning)


import segmentation_models_pytorch as smp
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from vitsegh.config import TrainingConfig, config_to_yaml, get_model_config
from vitsegh.constants import EXTRACTED_BANDS
from vitsegh.dataloader import get_dataloader
from vitsegh.models.segformer import SegFormer
from vitsegh.train_utils.binary import binary_segmentation_train
from vitsegh.train_utils.utils import get_loss, get_lr_scheduler, get_model
from vitsegh.utils.distributed import cleanup, setup
from vitsegh.utils.path import get_experiment_outpath


def main(rank: int, world_size: int, config: TrainingConfig) -> None:
    """
    Train SegFormer on EUSegSatellite dataset
    """
    setup(rank, world_size)
    print(f"Rank: {rank} - World size: {world_size}")

    comment = f"dim={config.image_size}_epochs={config.epochs}_lr={config.learning_rate}_bs={config.batch_size}"
    if config.weight_filepath:
        comment += "_checkpoint"
    config.output_path = get_experiment_outpath(
        config.output_path, config.model, comment=comment
    )
    if rank == 0:
        writer = SummaryWriter(config.output_path)
    else:
        writer = None

    device = torch.device("cuda", rank)
    dataloaders = get_dataloader(rank, world_size, config)
    dataset_len = len(dataloaders["train"].dataset) + len(dataloaders["val"].dataset)
    img, msk = dataloaders["train"].dataset[0]
    if rank == 0:
        print(f"Input Shape: {img.shape}, Output Shape: {msk.shape}")
        print(
            f"Length of training data: {len(dataloaders['train'].dataset)}, "
            + f"Length of validation data: {len(dataloaders['val'].dataset)}"
        )
        config_to_yaml(config, config.output_path / "config.yaml")

    model = get_model(config, device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=config.learning_rate)

    criterion = get_loss(config, rank)
    scheduler = get_lr_scheduler(optimizer, config, dataset_len)

    if config.weight_filepath:
        print(f"Loading pretrained weights from {config.weight_filepath.as_posix()}")
        checkpoint = torch.load(config.weight_filepath.as_posix())
        print(checkpoint.keys())
        ddp_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        #optimizer.load_state_dict(checkpoint["optimizer"])

    if config.num_classes == 1:
        binary_segmentation_train(
            config=config,
            dataloaders=dataloaders,
            rank=rank,
            optimizer=optimizer,
            model=ddp_model,
            criterion=criterion,
            lr_scheduler=scheduler,
            writer=writer,
        )
    else:
        raise ValueError("Only binary segmentation is supported at the moment")

    del model
    del dataloaders
    gc.collect()
    torch.cuda.empty_cache()
    cleanup()


def run():
    config = get_model_config()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, f"Requires at least 1 GPU to run, but got {n_gpus}"
    world_size = n_gpus

    mp.spawn(
        main,
        args=(
            world_size,
            config,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    run()
