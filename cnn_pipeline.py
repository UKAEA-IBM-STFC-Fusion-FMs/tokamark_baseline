import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp

# -------------------------------------------------------------------
# Repo-specific imports
# -------------------------------------------------------------------
from globals import REPO_ROOT, TOOLS_DIR
# from scripts.pipelines.globals import REPO_ROOT
from scripts.pipelines.utils.device_utils import get_device

from scripts.pipelines.utils.utils import (
    ComposeTransforms,
    initialize_dataloaders,
    initialize_model_datasets,
)

from scripts.pipelines.utils.preprocessing_utils import (
    initialize_datasets_and_metadata_for_task,
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
        default="/scripts/pipelines/configs/configs_task/task_1_reconstruction/config_task_1-1.yaml",
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
    # Initialize datasets and metadata
    # -------------------------------------------------------------------
    datasets_train_val_test, dict_metadata = initialize_datasets_and_metadata_for_task(
        config_task
    )

    # -------------------------------------------------------------------
    # CNN pipeline
    # -------------------------------------------------------------------

    model_specific_transform = ComposeTransforms(
        [
            TimeCNNTransform(dict_metadata),
        ]
    )

    datasets_cnn = initialize_model_datasets(
        datasets_train_val_test, dict_metadata, config_task, model_specific_transform
    )

    dataloaders_cnn = initialize_dataloaders(
        datasets_cnn, cnn_training_collate_fn, **config_cnn["dataloader_setting"]
    )

    cnn_model = create_cnn_architecture(
        dataloaders_cnn["train"], **config_cnn["cnn_settings"], verbose=True
    )

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    best_model_state, early_stop = loop_for_cnn_training(
        base_cnn_model=cnn_model,
        train_dataloader=dataloaders_cnn["train"],
        val_dataloader=dataloaders_cnn["val"],
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

    cnn_evaluation_per_shot(dataloaders_cnn["test"], config_task, cnn_model, config_cnn)

    cnn_save_traces_per_shot(
        dataloaders_cnn["test"], config_task, cnn_model, config_cnn, n_traces=10
    )
