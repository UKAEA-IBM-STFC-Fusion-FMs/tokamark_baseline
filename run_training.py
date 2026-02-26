import argparse
import yaml
import torch
from multiprocessing import cpu_count
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

try:
    from globals import REPO_ROOT
except:
    from .globals import REPO_ROOT

from utils import set_seed, seed_worker

from MAST_benchmark.tools.utils import get_device
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_benchmark.tasks import get_task_config, get_task_metadata
from MAST_benchmark.tools.transforms.compose_transform import (
    ComposeTransforms,
)
from MAST_benchmark.data import (
    initialize_MAST_dataset, 
    initialize_TokaMark_dataset,
)
from src.time_cnn_model import (
    create_cnn_architecture
)
from src.time_cnn_transform import (
    TimeCNNTransform,
)
from src.trainer import (
    cnn_collate_fn,
    # loop_for_cnn_training,
    # train_loop,
    BatchStepTrainer,
)

# Set device
device = get_device()
# print(f"Using device: {device}\n")


if __name__ == "__main__":

    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="task_4-5",
        help="The name of the task available in the benchmark",
    )
    parser.add_argument(
        "--config_cnn",
        type=str,
        default="/src/config/config_cnn_test.yaml",
        # default="/src/config/config_cnn_iterable_lr_4_work_4.yaml",
        help="Path to the model YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23, #200399 or 140801
        help="Path to the model YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Load Task YAML config
    config_task = get_task_config(args.task)    

    # Load CNN YAML config
    with open(REPO_ROOT + args.config_cnn, "r") as f:
        config_cnn = yaml.safe_load(f)
    print(config_cnn)

    SEED = args.seed
    set_seed(SEED)
    print(SEED)

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize task-specific metadata
    # ------------------------------------------------------------------------------------------------------------------
    # for training and validation: use stride of 0.005ms and 0.025ms
    if args.task in ["task_3-3",
                     "task_4-1", "task_4-2",
                     "task_4-3", "task_4-4", "task_4-5"]:              
        config_task["stride_window"] = 0.025
        # shuffle_buffer_size = 512
        shuffle_buffer_size = 2048
    else:
        config_task["stride_window"] = 0.005
        shuffle_buffer_size = 2048

    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize MAST datasets
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config_cnn["subset_of_shots"]
    )

    local_flag = config_cnn["local"]

    train_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        train_shots_,
        local_flag = local_flag,
        use_std_scaling = True,
        return_incomplete_shots=True,
        remove_outliers=True,
        verbose=False
    )
    val_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        val_shots_,
        local_flag = local_flag,
        use_std_scaling = True,
        return_incomplete_shots=True,
        remove_outliers=True,
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # CNN pipeline
    # ------------------------------------------------------------------------------------------------------------------

    model_specific_transform = ComposeTransforms(
        [
            TimeCNNTransform(dict_task_metadata | config_task),
        ]
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = initialize_TokaMark_dataset(
        train_MAST_dataset, dict_task_metadata, config_task, model_specific_transform, test_mode=True, 
    )
    train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=cnn_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            **config_cnn["dataloader_setting"],
            pin_memory=True,
            drop_last=True,
        )

    val_dataset = initialize_TokaMark_dataset(
        val_MAST_dataset, dict_task_metadata, config_task, model_specific_transform, test_mode=True
    )
    val_dataloader = DataLoader(
            dataset=val_dataset,
            collate_fn=cnn_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            **config_cnn["dataloader_setting"],
            pin_memory=True,
        )

    cnn_model = create_cnn_architecture(
        train_dataloader, **config_cnn["cnn_settings"], verbose=True
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------------------------------------------------------

    base = config_cnn["paths"]["data_output_directory"]

    base_model_dir = (
        REPO_ROOT
        + base
        + f"/{config_task['task_name']}/seed_{SEED}/"
    )

    trainer = BatchStepTrainer(
        model=cnn_model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        **config_cnn["training_args"],
        output_dir=base_model_dir,
        device=device,
        validate_every=100,  # validate every 100 batches
    )

    # Step through batches
    while trainer.step_batch():
        pass
