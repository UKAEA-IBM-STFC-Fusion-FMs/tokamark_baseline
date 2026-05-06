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


from src.multi_conv_mlp_model import (
    create_cnn_architecture
)
from src.multi_conv_lstm_model import (
    create_lstm_architecture
)
from src.model_transform import (
    ModelTransform_1,
    ModelTransform_2,
)
from src.trainer import (
    model_collate_fn,
)

from src.evaluator import (
    cnn_unstd_evaluation_per_shot,
)

from tokamark.tools.path import (
    RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE, 
    RANDOM_SPLIT_SIGNALS_STATS_FILE,
    TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE, 
    TEMPORAL_SPLIT_SIGNALS_STATS_FILE
    )

from MAST_tools.utils.path_utils import (
    RANDOM_SPLIT_OUTLIER_METADATA_FILE,
    TEMPORAL_SPLIT_OUTLIER_METADATA_FILE,
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
        default="task_2-1",
        help="The name of the task available in the benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/src/config/config_model_test.yaml",
        help="Path to the model YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Specified seed"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        help="Model type to train."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        help="Splitting used."
    )
    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Some configuration tasks
    # ------------------------------------------------------------------------------------------------------------------

    # Load Task YAML config
    config_task = get_task_config(task_name=args.task)

    # Load CNN YAML config
    with open(REPO_ROOT + args.config, "r") as f:
        config = yaml.safe_load(f)
    
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
    # Load correct settings
    # ------------------------------------------------------------------------------------------------------------------

    if args.split == 'random':

        DATA_SPLIT = RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE
        OUTLIER_FILE = RANDOM_SPLIT_OUTLIER_METADATA_FILE
        SIGNAL_STATS = RANDOM_SPLIT_SIGNALS_STATS_FILE

    elif args.split == 'temporal':

        DATA_SPLIT = TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE
        OUTLIER_FILE = TEMPORAL_SPLIT_OUTLIER_METADATA_FILE
        SIGNAL_STATS = TEMPORAL_SPLIT_SIGNALS_STATS_FILE
    
    else:
        print('SPLIT UNKNOWN')

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize MAST datasets
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config["subset_of_shots"],        
        shuffle=True,
        data_splits_file_path = DATA_SPLIT,
    )

    local_flag = config["local"]

    test_MAST_dataset = initialize_MAST_dataset( 
        config_task=config_task,
        shots_list=test_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        stats_metadata_file_path=SIGNAL_STATS,
        use_nan_filling=False,
        return_incomplete_shots=True,
        remove_outliers=True,
        outlier_metadata_file=OUTLIER_FILE,
        remove_bad_efit_rating=True,
        store_manager_settings=config["store_manager_settings"],
        verbose=False
    )

    # ------------------------------------------------------------------------------------------------------------------
    # CNN pipeline
    # ------------------------------------------------------------------------------------------------------------------
    
    model_specific_transform = ComposeTransforms(
        [
            ModelTransform_1(dict_task_metadata | config_task),
            ModelTransform_2(dict_task_metadata | config_task),
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
            collate_fn=model_collate_fn,
            **config["dataloader_setting"],
            pin_memory=True,
        )    
    test_dataloader_vizu = DataLoader(
            dataset=test_dataset,
            collate_fn=model_collate_fn,
            batch_size=1,
            num_workers=0,
        )
    # ------------------------------------------------------------------------------------------------------------------
    # Initialize Model
    # ------------------------------------------------------------------------------------------------------------------

    if args.model == 'cnn':

        model = create_cnn_architecture(
            dataloader_=test_dataloader,
            dict_metadata = dict_task_metadata | config_task,
            verbose=True
        )

    elif args.model == 'lstm':

        model = create_lstm_architecture(
            dataloader_=test_dataloader,
            dict_metadata = dict_task_metadata | config_task,
            verbose=True
        )
         
    else:
        print('Model Unknown')

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------

    base = config["paths"]["data_output_directory"]

    base_model_dir = (
        REPO_ROOT
        + base
        + f"/{args.split}/{args.model}/{config_task['task_name']}/seed_{SEED}/"
    )

    # -------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------

    results_dir = REPO_ROOT + f"/results/{args.split}/{args.model}/seed_{SEED}/"

    # cnn_safety_vizu_per_shot(test_dataloader_vizu, config_task, cnn_model, base_model_dir, n_shot_to_plot=3)
    accumulator = WindowMetricsAccumulator(args.task)
    cnn_unstd_evaluation_per_shot(test_dataloader, config_task, model, base_model_dir, accumulator)
    
    compute_metrics(
        task=args.task,
        output_dir=results_dir,
        window_metrics_accumulator=accumulator,
        save_windows_metrics=True,
        save_task_metrics=True,
    )
