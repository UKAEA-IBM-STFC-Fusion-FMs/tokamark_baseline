import argparse
from torch.utils.data.dataloader import DataLoader
from typing import Any
import yaml
import torch
from multiprocessing import cpu_count
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

try:
    from globals import REPO_ROOT
except ImportError:
    from .globals import REPO_ROOT

from utils import set_seed, seed_worker

from tokamark.tools.utils import get_device
from tokamark.data_split import get_train_test_val_shots
from tokamark.tasks import get_task_config, get_task_metadata
from tokamark.tools.transforms.compose_transform import (
    ComposeTransforms,
)
from tokamark.data import (
    initialize_MAST_dataset, 
    initialize_TokaMark_dataset,
)
from src.lstm_model import (
    create_lstm_architecture
)
from src.lstm_transform import (
    LstmTransform_1,
    LstmTransform_2,
)
from src.trainer import (
    lstm_collate_fn,
    BatchStepTrainer,
)


# ----------------------------------------------------------------------------------------------------------------------

# Set device
device = get_device()
# print(f"Using device: {device}\n")


# ======================================================================================================================
if __name__ == "__main__":

    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method(method="spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="task_1-1",
        help="The name of the task available in the benchmark"
    )
    parser.add_argument(
        "--config_lstm",
        type=str,
        default="/src/config_lstm/config_lstm_test.yaml",
        # default="/src/config/config_cnn_iterable_lr_4_work_4.yaml",
        help="Path to the model YAML config file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Specified seed."
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=100,
        help="Number of batches at which validation is performed."
    )

    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Some configuration tasks
    # ------------------------------------------------------------------------------------------------------------------

    # Load Task YAML config
    config_task = get_task_config(task_name=args.task)

    # Load CNN YAML config
    with open(REPO_ROOT + args.config_lstm, "r") as f:
        config_lstm = yaml.safe_load(f)
    print(config_lstm)

    SEED = args.seed
    set_seed(SEED)
    print(SEED)

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize task-specific metadata
    # ------------------------------------------------------------------------------------------------------------------

    # For training and validation: use stride of 0.005ms and 0.025ms
    if args.task in ["task_3-3",
                     "task_4-1", "task_4-2",
                     "task_4-3", "task_4-4", "task_4-5"]:              
        config_task["stride_window"] = 0.025
        shuffle_buffer_size = 2048  # Typical options: 512, 2048
    else:
        config_task["stride_window"] = 0.005
        shuffle_buffer_size = 2048

    dict_task_metadata = get_task_metadata(
        config_task=config_task,
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize MAST datasets
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config_lstm["subset_of_shots"]
    )

    local_flag = config_lstm["local"]

    train_MAST_dataset = initialize_MAST_dataset( 
        config_task=config_task,
        shots_list=train_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        # use_nan_filling=False,
        use_nan_filling=True,
        return_incomplete_shots=True,
        remove_outliers=True,
        remove_bad_efit_rating=True,
        verbose=False
    )
    val_MAST_dataset = initialize_MAST_dataset( 
        config_task=config_task,
        shots_list=val_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        # use_nan_filling=False,
        use_nan_filling=True,
        return_incomplete_shots=True,
        remove_outliers=True,
        remove_bad_efit_rating=True,
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # CNN pipeline
    # ------------------------------------------------------------------------------------------------------------------

    model_specific_transform = ComposeTransforms(
        [
            LstmTransform_1(dict_task_metadata | config_task),
            LstmTransform_2(dict_task_metadata | config_task),
        ]
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = initialize_TokaMark_dataset(
        dataset=train_MAST_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True
    )
    train_dataloader: DataLoader[Any] = DataLoader(
            dataset=train_dataset,
            collate_fn=lstm_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            **config_lstm["dataloader_setting"],
            pin_memory=True,
            drop_last=True
        )

    val_dataset = initialize_TokaMark_dataset(
        dataset=val_MAST_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True
    )
    val_dataloader = DataLoader(
            dataset=val_dataset,
            collate_fn=lstm_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            **config_lstm["dataloader_setting"],
            pin_memory=True
        )

    lstm_model = create_lstm_architecture(
        dataloader_=train_dataloader,
        dict_metadata = dict_task_metadata | config_task,
        **config_lstm["lstm_settings"],
        verbose=True
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------------------------------------------------------

    base = config_lstm["paths"]["data_output_directory"]

    base_model_dir = (
        REPO_ROOT
        + base
        + f"/{config_task['task_name']}/seed_{SEED}/"
    )

    trainer = BatchStepTrainer(
        model=lstm_model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        **config_lstm["training_args"],
        output_dir=base_model_dir,
        device=device,
        validate_every=args.validate_every,  # Validate every 100 batches by default.
    )

    # Step through batches
    while trainer.step_batch():
        # A single pass update is performed as long as `trainer.step_batch()` is True.
        pass
