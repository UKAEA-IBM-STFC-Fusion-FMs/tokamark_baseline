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


# ----------------------------------------------------------------------------------------------------------------------
# CNN SAFETY VISUALIZATION LOOP
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def cnn_safety_vizu_per_shot(  # NOSONAR - Ignore cognitive complexity
    test_dataloader,
    config_task,
    cnn_model,
    output_dir,
    n_shot_to_plot=3
):
    """Plot some shots CNN per shot/window and save incremental RMSEs to CSV."""

    # === Setup paths ===
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
    counter = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            
            if counter >= n_shot_to_plot:
                break

            if batch is None:
                continue

            counter += 1

            shot_id, window_id, x_test, y_test = batch

            plot_dir = output_dir + f"safety_vizu/shot_{shot_id[0]}/"
            os.makedirs(plot_dir, exist_ok=True)

            # Move inputs and labels to device
            x_test = [arr.to(torch.float32).to(device) for arr in x_test]
            y_test = [arr.to(torch.float32).to(device) for arr in y_test]

            # Model prediction
            y_pred = cnn_model(*x_test)

            # Make sure y_pred is list-like
            if not isinstance(y_pred, (list, tuple)):
                y_pred = [y_pred]

            # === Compute RMSEs per feature ===
            for i, feature_name in enumerate(feature_names):

                print(f"{feature_name[0]}-{feature_name[1]}")

                y_t = (
                    y_test[i]
                    .detach()
                    .cpu()
                    .squeeze(1)
                    .numpy()
                )
                y_p = (
                    y_pred[i]
                    .detach()
                    .cpu()
                    .squeeze(1)
                    .numpy()
                )

                mean = dict_task_metadata['output'][f"{feature_name[0]}-{feature_name[1]}"]['mean']
                std = dict_task_metadata['output'][f"{feature_name[0]}-{feature_name[1]}"]['std']

                unstd_y_t = y_t*std + mean
                unstd_y_p = y_p*std + mean 

                print('unstd_y_t.shape', unstd_y_t.shape)
                print('unstd_y_t.ndim', unstd_y_t.ndim)

                if unstd_y_t.ndim == 1:

                    plt.figure(figsize=(10, 5))

                    plt.plot(unstd_y_t, label="True")
                    plt.plot(unstd_y_p, '--', label="Pred")

                    plt.title("True vs Predicted")
                    plt.xlabel("Sample")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)

                elif (unstd_y_t.ndim == 2) and (unstd_y_t.shape[1] == 2):

                    print('hey')

                    plt.figure(figsize=(10, 5))

                    plt.plot(unstd_y_t[:, 0], label="True")
                    plt.plot(unstd_y_p[:, 0], '--', label="Pred")
                
                    plt.plot(unstd_y_t[:, 1], label="True")
                    plt.plot(unstd_y_p[:, 1], '--', label="Pred")

                    plt.title("True vs Predicted")
                    plt.xlabel("Sample")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)
                
                elif (unstd_y_t.ndim == 2) and (unstd_y_t.shape[1] != 2):

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    im0 = axes[0].imshow(unstd_y_t)
                    axes[0].set_title("True")
                    axes[0].set_xlabel("Sample")
                    axes[0].set_ylabel("Value")
                    plt.colorbar(im0, ax=axes[0])

                    im1 = axes[1].imshow(unstd_y_p)
                    axes[1].set_title("Predicted")
                    axes[1].set_xlabel("Sample")
                    axes[1].set_ylabel("Value")
                    plt.colorbar(im1, ax=axes[1])

                    plt.suptitle("True vs Predicted")
                    plt.tight_layout()
                    plt.show()
                
                elif (unstd_y_t.ndim == 3) and (unstd_y_t.shape[2] == 2):
                    
                    n_dim = unstd_y_t.shape[-1]
                    print(unstd_y_t.shape)

                    plt.figure(figsize=(10, 5))

                    # Line plots
                    colors = plt.cm.tab10.colors
                    
                    for k in range(n_dim): 
                        c = colors[k % len(colors)]
                        plt.plot(unstd_y_t[:, 0, k], label=f"True {k+1}", color=c)
                        plt.plot(unstd_y_p[:, 0, k], '--', label=f"Pred {k+1}", color=c)

                    plt.title("True vs Predicted")
                    plt.xlabel("Sample")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)
                
                elif unstd_y_t.ndim == 4:

                    n_time = np.linspace(0, unstd_y_p.shape[0] - 1, 5, dtype=int)

                    fig, axes = plt.subplots(3, len(n_time), figsize=(30, 15))

                    # Handle case where n_time = 1 (axes is 1D instead of 2D)
                    if len(n_time) == 1:
                        axes = axes.reshape(1, -1)

                    for i, t in enumerate(n_time):
 
                        print(t)

                        y_p_t = unstd_y_p[t, 0]
                        y_t_t = unstd_y_t[t, 0]

                        # Heatmap
                        error = y_p_t - y_t_t

                        # Subplot 1: True values heatmap
                        im1 = axes[0, i].imshow(y_t_t.T, aspect="auto", cmap="viridis")
                        fig.colorbar(im1, ax=axes[0, i], label="True Value")
                        axes[0, i].set_title(f"True Values (t={t})")
                        axes[0, i].set_xlabel("Sample")
                        axes[0, i].set_ylabel("Output Dimension")

                        # Subplot 2: Predicted values heatmap
                        im2 = axes[1, i].imshow(y_p_t.T, aspect="auto", cmap="viridis")
                        fig.colorbar(im2, ax=axes[1, i], label="Predicted Value")
                        axes[1, i].set_title(f"Predicted Values (t={t})")
                        axes[1, i].set_xlabel("Sample")
                        axes[1, i].set_ylabel("Output Dimension")

                        # Subplot 3: Error heatmap
                        im3 = axes[2, i].imshow(error.T, aspect="auto", cmap="coolwarm")
                        fig.colorbar(im3, ax=axes[2, i], label="Prediction Error")
                        axes[2, i].set_title(f"Prediction Error (t={t})")
                        axes[2, i].set_xlabel("Sample")
                        axes[2, i].set_ylabel("Output Dimension")

                else:

                    plt.figure(figsize=(10, 5))

                    flat_y_p = unstd_y_p.reshape(len(shot_id), -1)
                    flat_y_t = unstd_y_t.reshape(len(shot_id), -1)

                    # Heatmap
                    error = flat_y_p - flat_y_t

                    # Subplot 1: True values heatmap
                    plt.subplot(1, 3, 1)
                    im1 = plt.imshow(flat_y_t.T, aspect="auto", cmap="viridis")
                    plt.colorbar(im1, label="True Value")
                    plt.title("True Values")
                    plt.xlabel("Sample")
                    plt.ylabel("Output Dimension")

                    # Subplot 2: Predicted values heatmap
                    plt.subplot(1, 3, 2)
                    im2 = plt.imshow(flat_y_p.T, aspect="auto", cmap="viridis")
                    plt.colorbar(im2, label="Predicted Value")
                    plt.title("Predicted Values")
                    plt.xlabel("Sample")
                    plt.ylabel("Output Dimension")

                    # Subplot 3: Error heatmap
                    plt.subplot(1, 3, 3)
                    im3 = plt.imshow(error.T, aspect="auto", cmap="coolwarm")
                    plt.colorbar(im3, label="Prediction Error")
                    plt.title("Prediction Error")
                    plt.xlabel("Sample")
                    plt.ylabel("Output Dimension")

                plt.tight_layout()
                plt.savefig(
                    plot_dir + f"prediction_plot_{feature_name[0]}-{feature_name[1]}.png",
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close() 

    print(f"Visualization saved. Go check in: {output_dir}")
