import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# -------------------------------------------------------------------
# Repo-specific imports
# -------------------------------------------------------------------
from globals import REPO_ROOT, TOOLS_DIR

from scripts.pipeline_tools.utils import get_device
from scripts.pipeline_tools.utils import get_train_test_val_shots

from scripts.pipeline_tools.transforms.compose_transform import (
    ComposeTransforms,
)

from scripts.pipeline_tools.initialize_MAST_dataset import (
    initialize_MAST_dataset,
)

from scripts.pipeline_tools.get_task_metadata import (
    get_task_metadata,
)

from scripts.pipeline_tools.initialize_model_dataset import (
    initialize_model_dataset,
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
        "--config_task",
        type=str,
        default="/configs_task/task_1_reconstruction/config_task_1-1.yaml",
        help="Path to the task YAML config file",
    )
    parser.add_argument(
        "--config_cnn",
        type=str,
        default="/config_cnn_reconstruction.yaml",
        help="Path to the model YAML config file",
    )
    args, _ = parser.parse_known_args()

    # Load Task YAML config
    with open(TOOLS_DIR + args.config_task, "r") as f:
        config_task = yaml.safe_load(f)

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
        max_index=config_task["subset_of_shots"]
    )

    train_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        train_shots_,
        use_std_scaling = True,
        return_incomplete_shots=True
    )

    val_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        val_shots_,
        use_std_scaling = True,
        return_incomplete_shots=True
    )

    test_MAST_dataset = initialize_MAST_dataset( 
        config_task,
        test_shots_,
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
