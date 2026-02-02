import os
import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# -------------------------------------------------------------------
# Repo-specific imports
# -------------------------------------------------------------------
try:
    from globals import REPO_ROOT
except:
    from .globals import REPO_ROOT

from MAST_benchmark.tools.utils import get_device
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_benchmark.tasks import get_task_config, get_task_metadata
from MAST_benchmark.tools.transforms.compose_transform import (
    ComposeTransforms,
)
from MAST_benchmark.data import (
    initialize_MAST_dataset, initialize_model_dataset
)

from timecnn_transform_cutting_input import (
    TimeCNNTransform_cutting,
)

from cnn_utils import (
    cnn_filled_training_collate_fn,
    create_cnn_v5_architecture,
    loop_for_cnn_training,
)

# Set device
device = get_device()
# print(f"Using device: {device}\n")


if __name__ == "__main__":
    print(f"Number of available CPU cores: {cpu_count()}\n")
    mp.set_start_method("spawn", force=True)

    # -------------------------------------------------------------------
    # Argument parsing
    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        # default="task_1-1",
        default="task_4-1",
        help="The name of the task available in the benchmark",
    )
    parser.add_argument(
        "--config_cnn",
        type=str,
        default="/config/config_cnn_test.yaml",
        help="Path to the model YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Load Task YAML config
    config_task = get_task_config(args.task)    

    # Load CNN YAML config
    with open(REPO_ROOT + args.config_cnn, "r") as f:
        config_cnn = yaml.safe_load(f)

    # -------------------------------------------------------------------
    # Initialize task-specific metadata
    # -------------------------------------------------------------------

    # for training and validation: use stride of 0.005ms
    config_task["stride_window"] = 0.005

    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

    # -------------------------------------------------------------------
    # Initialize MAST datasets
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # CNN pipeline
    # -------------------------------------------------------------------

    model_specific_transform = ComposeTransforms(
        [
            TimeCNNTransform_cutting(dict_task_metadata | config_task),
        ]
    )

    train_dataset = initialize_model_dataset(
        train_MAST_dataset, dict_task_metadata, config_task, model_specific_transform, test_mode=True
    )
    train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=cnn_filled_training_collate_fn,
            **config_cnn["dataloader_setting"],
            pin_memory=True
        )

    val_dataset = initialize_model_dataset(
        val_MAST_dataset, dict_task_metadata, config_task, model_specific_transform, test_mode=True
    )
    val_dataloader = DataLoader(
            dataset=val_dataset,
            collate_fn=cnn_filled_training_collate_fn,
            **config_cnn["dataloader_setting"],
            pin_memory=True
        )

    cnn_model = create_cnn_v5_architecture(
        train_dataloader, **config_cnn["cnn_settings"], verbose=True
    )

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------

    base = config_cnn["paths"]["data_output_directory"]

    print(config_cnn)

    base_model_dir = (
        REPO_ROOT
        + base
        + f"/{config_task['task_name']}"
        + "/model_v5_filled_cutting_input/"
    )
    
    counter = 1
    model_dir = base_model_dir.rstrip("/") + f"/run_{counter}/"
    # If folder exists → create new version
    # while os.path.exists(model_dir):
    #     counter += 1
    #     model_dir = base_model_dir.rstrip("/") + f"/run_{counter}/"
    # Create directory
    os.makedirs(model_dir, exist_ok=True)
    # print(f"Saving model to: {model_dir}")

    best_model_state, early_stop = loop_for_cnn_training(
        base_cnn_model=cnn_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **config_cnn["training_args"],
        output_dir=model_dir,
        verbose=True,
    )

    # -------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------

    # cnn_unstd_evaluation_per_shot(test_dataloader, config_task, cnn_model, model_dir)

    # cnn_save_traces_per_shot(
    #     test_dataloader, config_task, cnn_model, config_cnn, n_traces=10
    # )


