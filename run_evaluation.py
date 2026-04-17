import argparse
import yaml
from multiprocessing import cpu_count
import torch.multiprocessing as mp

from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------------------------------------------------
# Repo-specific imports
# ----------------------------------------------------------------------------------------------------------------------

try:
    from globals import REPO_ROOT
except ImportError:
    from .globals import REPO_ROOT

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

from tokamark.evaluator import (
    WindowMetricsAccumulator,
    compute_metrics,
)


from src.time_cnn_model import (
    create_cnn_architecture
)

from src.time_cnn_transform import (
    TimeCNNTransform,
)

from src.trainer import (
    cnn_collate_fn,
)

from src.evaluator import (
    # cnn_safety_vizu_per_shot,
    cnn_unstd_evaluation_per_shot,
)


# ----------------------------------------------------------------------------------------------------------------------

# Set device
device = get_device()
# print(f"Using device: {device}\n")


# ======================================================================================================================
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
        default="task_1-1",
        help="The name of the task available in the benchmark",
    )
    parser.add_argument(
        "--config_cnn",
        type=str,
        default="/src/config/config_cnn_iterable_lr_4_work_4.yaml",
        help="Path to the model YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Specified seed"
    )
    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Some configuration tasks
    # ------------------------------------------------------------------------------------------------------------------

    # Load Task YAML config
    config_task = get_task_config(task_name=args.task)

    # Load CNN YAML config
    with open(REPO_ROOT + args.config_cnn, "r") as f:
        config_cnn = yaml.safe_load(f)
    
    SEED = args.seed
    print(SEED)
    
    # ------------------------------------------------------------------------------------------------------------------
    # Initialize task-specific metadata
    # ------------------------------------------------------------------------------------------------------------------

    dict_task_metadata = get_task_metadata(
        config_task=config_task,
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize MAST datasets
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config_cnn["subset_of_shots"]
    )

    local_flag = config_cnn["local"]

    test_MAST_dataset = initialize_MAST_dataset( 
        config_task=config_task,
        shots_list=test_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
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
    
    test_dataset = initialize_TokaMark_dataset(
        dataset=test_MAST_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True,
        shuffle_windows=False
    )
    
    if test_dataset is None:
        raise ValueError("Failed to initialize test dataset. test_MAST_dataset may be None or invalid.")
    
    test_dataloader = DataLoader(
            dataset=test_dataset,
            collate_fn=cnn_collate_fn,
            **config_cnn["dataloader_setting"],
            pin_memory=True,
        )    
    test_dataloader_vizu = DataLoader(
            dataset=test_dataset,
            collate_fn=cnn_collate_fn,
            batch_size=1,
            num_workers=0,
        )

    cnn_model = create_cnn_architecture(
        dataloader_=test_dataloader_vizu,
        **config_cnn["cnn_settings"],
        verbose=True
    )

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------

    base = config_cnn["paths"]["data_output_directory"]

    print(config_cnn)
    base_model_dir = (
        REPO_ROOT
        + base
        + f"/{config_task['task_name']}/seed_{SEED}/"
    )

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------

    results_dir = REPO_ROOT + f"/results_NEW_vfinal_MR_RERUN/seed_{SEED}/"

    # cnn_safety_vizu_per_shot(test_dataloader_vizu, config_task, cnn_model, base_model_dir, n_shot_to_plot=3)
    accumulator = WindowMetricsAccumulator(args.task)
    cnn_unstd_evaluation_per_shot(test_dataloader, config_task, cnn_model, base_model_dir, accumulator)
    
    compute_metrics(
        task=args.task,
        output_dir=results_dir,
        window_metrics_accumulator=accumulator,
        save_windows_metrics=True,
        save_task_metrics=True,
    )
