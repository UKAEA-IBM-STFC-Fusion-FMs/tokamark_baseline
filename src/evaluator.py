import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from tokamark.tasks import get_task_metadata
from tokamark.tools.utils import get_device


# ----------------------------------------------------------------------------------------------------------------------

# Set device
device = get_device()


# ----------------------------------------------------------------------------------------------------------------------
# CNN EVALUATION LOOP
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def cnn_unstd_evaluation_per_shot(
    test_dataloader,
    config_task,
    cnn_model,
    output_dir,
    accumulator
):
    """Evaluate CNN per shot/window and save incremental RMSEs to CSV."""

    print("in cnn_unstd_evaluation_per_shot")

    best_model_path = output_dir + "best_model.pt"

    # Load best model
    cnn_model.load_state_dict(torch.load(best_model_path, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

    feature_names = config_task["sources_and_signals"].get("output_name", [])
    
    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

    # === Evaluation loop ===
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch is None:
                continue

            shot_id, window_id, x_test, y_test = batch

            # Move inputs and labels to device
            x_test = [arr.to(torch.float32).to(device) for arr in x_test]
            y_test = [arr.to(torch.float32).to(device) for arr in y_test]

            # Model prediction
            y_pred = cnn_model(*x_test)

            # Make sure y_pred is list-like
            if not isinstance(y_pred, (list, tuple)):
                y_pred = [y_pred]

            for i, feature_name in enumerate(feature_names):

                y_t = (
                    y_test[i]
                    .detach()
                    .cpu()
                    .squeeze(1)
                    .reshape(len(shot_id), -1)
                    .numpy()
                )
                y_p = (
                    y_pred[i]
                    .detach()
                    .cpu()
                    .squeeze(1)
                    .reshape(len(shot_id), -1)
                    .numpy()
                )

                mean = dict_task_metadata['output'][f"{feature_name[0]}-{feature_name[1]}"]['mean']
                std = dict_task_metadata['output'][f"{feature_name[0]}-{feature_name[1]}"]['std']

                unstd_y_t = y_t*std + mean
                unstd_y_p = y_p*std + mean 

                accumulator.add_batch(
                    y_target=np.float128(unstd_y_t),
                    y_pred=np.float128(unstd_y_p),
                    shot_ids=shot_id,
                    window_indices=window_id,
                    feature_name=f"{feature_name[0]}-{feature_name[1]}",
                )

    print("💅🏼 UNSTD Evaluation done. RMSEs and MSEs saved (incrementally).")