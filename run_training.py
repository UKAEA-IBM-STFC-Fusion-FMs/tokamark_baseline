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
    BatchStepTrainer,
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
# print(f"Using device: {device} and pin_memory is {torch.cuda.is_available()}\n")


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
        "--config",
        type=str,
        default="/src/config/config_model_test.yaml",
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
    set_seed(SEED)

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

    if args.task in [
        "task_1-1",
        "task_1-2", 
        "task_1-3",
        "task_2-1",
        "task_2-2", 
        "task_2-3",        
        "task_3-1",
        "task_3-2", 
        "task_3-3",
        "task_4-3"
        ]:
        config['dataloader_setting']['num_workers'] = 8
    
    elif args.task in [
        "task_4-1",
        "task_4-2", 
        "task_4-4", 
        "task_4-5"]:
        config['dataloader_setting']['num_workers'] = 4

    else:
        print('Task Unknown')

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
        raise ValueError('Split Unknkwn')
        

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize MAST datasets
    # ------------------------------------------------------------------------------------------------------------------

    train_shots_, test_shots_, val_shots_ = get_train_test_val_shots(
        max_index=config["subset_of_shots"],
        # shuffle=True,
        data_splits_file_path = DATA_SPLIT        
    )

    local_flag = config["local"]

    train_MAST_dataset = initialize_MAST_dataset( 
        config_task=config_task,
        shots_list=train_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        stats_metadata_file_path=SIGNAL_STATS,
        use_nan_filling=False,
        remove_outliers=True,
        outlier_metadata_file=OUTLIER_FILE,
        remove_bad_efit_rating=True,
        store_manager_settings=config["store_manager_settings"],
        verbose=False
    )
    val_MAST_dataset = initialize_MAST_dataset( 
        config_task=config_task,
        shots_list=val_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        stats_metadata_file_path=SIGNAL_STATS,
        use_nan_filling=False,
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
            collate_fn=model_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            **config["dataloader_setting"],
            pin_memory=torch.cuda.is_available(),
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
            collate_fn=model_collate_fn,
            worker_init_fn=seed_worker,
            generator=g,
            **config["dataloader_setting"],
            pin_memory=torch.cuda.is_available()
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Initialize Model
    # ------------------------------------------------------------------------------------------------------------------

    if args.model == 'cnn':

        model = create_cnn_architecture(
            dataloader_=train_dataloader,
            dict_metadata = dict_task_metadata | config_task,
            verbose=False
        )

    elif args.model == 'lstm':

        model = create_lstm_architecture(
            dataloader_=train_dataloader,
            dict_metadata = dict_task_metadata | config_task,
            verbose=False
        )
    else:
        raise ValueError('Model Unknown')
        
    # ------------------------------------------------------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------------------------------------------------------

    base = config["paths"]["data_output_directory"]

    base_model_dir = (
        REPO_ROOT
        + base
        + f"/{args.split}/{args.model}/{config_task['task_name']}/seed_{SEED}/"
    )

    trainer = BatchStepTrainer(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        **config["training_args"],
        output_dir=base_model_dir,
        device=device,
        validate_every=args.validate_every,  # Validate every 100 batches by default.
    )

    # Step through batches
    while trainer.step_batch():
        # A single pass update is performed as long as `trainer.step_batch()` is True.
        pass
