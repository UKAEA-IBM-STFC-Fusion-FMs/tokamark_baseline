import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# -------------------------------------------------------------------
# Repo-specific imports
# -------------------------------------------------------------------
from globals import REPO_ROOT, TOOLS_DIR

from MAST_benchmark.tools.utils import get_device
from MAST_benchmark.data_split import get_train_test_val_shots
from MAST_benchmark.tasks import get_task_config, get_task_metadata
from MAST_benchmark.tools.transforms.compose_transform import (
    ComposeTransforms,
)
from MAST_benchmark.data import (
    initialize_MAST_dataset, initialize_model_dataset
)

from timecnn_transform import (
    TimeCNNTransform,
)
from cnn_utils import (
    cnn_training_collate_fn,
    create_cnn_architecture,
    loop_for_cnn_training,
    cnn_evaluation_per_shot,
    cnn_save_traces_per_shot,
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
        default="task_1-1",
        help="The name of the task available in the benchmark",
    )
    parser.add_argument(
        "--config_cnn",
        type=str,
        default="/config_cnn_reconstruction.yaml",
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
        return_incomplete_shots=True
    )

    val_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        val_shots_,
        local_flag = local_flag,
        use_std_scaling = True,
        return_incomplete_shots=True
    )

    test_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        test_shots_,
        local_flag = local_flag,
        use_std_scaling = True,
        return_incomplete_shots=True
    )

    # -------------------------------------------------------------------
    # CNN pipeline
    # -------------------------------------------------------------------

    model_specific_transform = ComposeTransforms(
        [
            TimeCNNTransform(dict_task_metadata),
        ]
    )

    train_dataset = initialize_model_dataset(
        train_MAST_dataset, dict_task_metadata, config_task, model_specific_transform
    )
    train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=cnn_training_collate_fn,
            **config_cnn["dataloader_setting"]
        )

    val_dataset = initialize_model_dataset(
        val_MAST_dataset, dict_task_metadata, config_task, model_specific_transform
    )
    val_dataloader = DataLoader(
            dataset=val_dataset,
            collate_fn=cnn_training_collate_fn,
            **config_cnn["dataloader_setting"]
        )
    
    test_dataset = initialize_model_dataset(
        test_MAST_dataset, dict_task_metadata, config_task, model_specific_transform
    )
    test_dataloader = DataLoader(
            dataset=test_dataset,
            collate_fn=cnn_training_collate_fn,
            **config_cnn["dataloader_setting"]
        )

    cnn_model = create_cnn_architecture(
        train_dataloader, **config_cnn["cnn_settings"], verbose=True
    )

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    best_model_state, early_stop = loop_for_cnn_training(
        base_cnn_model=cnn_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **config_cnn["training_args"],
        output_dir=REPO_ROOT
        + config_cnn["paths"]["data_output_directory"] 
        + config_task["task_name"]
        + "/",
        verbose=True,
    )

    # -------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------

    cnn_evaluation_per_shot(test_dataloader, config_task, cnn_model, config_cnn)

    cnn_save_traces_per_shot(
        test_dataloader, config_task, cnn_model, config_cnn, n_traces=10
    )
