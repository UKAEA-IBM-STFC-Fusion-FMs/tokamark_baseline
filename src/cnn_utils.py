from torchinfo import summary

# Set device
from MAST_benchmark.tools.utils import get_device

device = get_device()
# print(f"Using device: {device}\n")

import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from MAST_benchmark.tasks import get_task_metadata

# from time_cnn_model import MultiBranchTimeCNNModel
from time_cnn_model_v5 import MultiBranchTimeCNNModel_v5


# ----------------------------------------------------------------------------------------------------------------------
# COLLATE FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def cnn_filled_training_collate_fn(batch, verbose=True):

    # proc = psutil.Process(os.getpid())
    # mem = proc.memory_info().rss / (1024**2)
    # print(f"[Worker PID={proc.pid}] Memory={mem:.2f} MB")

    # Flatten the batch of lists into a single list
    full_flattened_batch = [
        (item["shot_id"], item["window_index"], 
        [np.nan_to_num(np.array(x), nan=0.0) for x in item["x"]],
        [np.nan_to_num(np.array(y), nan=0.0) for y in item["y"]])
        for sublist in batch
        for item in sublist
    ]

    if verbose:
        print(
            f"Collating batch of size = {len(batch)} shots to N = {len(full_flattened_batch)}"
        )

    return default_collate(full_flattened_batch) if (len(full_flattened_batch) > 0) else None


# ----------------------------------------------------------------------------------------------------------------------
# CNN TRAINING
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def create_cnn_v5_architecture(dataloader_, D, verbose=False):
    if verbose:
        print("\n\n----------MODEL INITIALIZATION V4 CNN----------\n")

    for l in range(len(dataloader_.dataset)):
        try:
            # Get the generator from __getitem__
            windows_gen = dataloader_.dataset[l]  # this is now a generator
            first_window = next(windows_gen)  # get the first yielded window
            input_shapes = [arr.shape for arr in first_window["x"]]
            output_shape = [arr.shape for arr in first_window["y"]]

            if verbose:
                print(f"Shot {dataloader_.dataset.get_shot_id(l)}")
                print(f"input_shapes: {input_shapes}")
                print(f"output_shape: {output_shape}")

            break  # stop after first successful shot

        except Exception as e:
            print(
                f"Skipping {dataloader_.dataset.get_shot_id(l)} because shot not trainable: {e}"
            )
            continue

    cnn_model = MultiBranchTimeCNNModel_v5(input_shapes, output_shape, D).to(device)

    input_size = [ (2,) + shape for shape in input_shapes ]
    summary(cnn_model, input_size=input_size)

    return cnn_model

# ----------------------------------------------------------------------------------------------------------------------
class MultiOutputMSELoss(nn.Module):
    def __init__(self, reduction="mean", weights=None):
        super().__init__()
        self.reduction = reduction
        # self.weights = weights  # e.g. [1.0, 0.5, 0.1, 2.0]

    def forward(self, y_preds, y_trues):
        assert len(y_preds) == len(y_trues), "Mismatch in number of outputs"
        losses = []
        for i, (yp, yt) in enumerate(zip(y_preds, y_trues)):
            assert yp.shape == yt.shape, (
                f"Shape mismatch at output {i}: {yp.shape} vs {yt.shape}"
            )
            l = F.mse_loss(yp, yt, reduction=self.reduction)
            # if self.weights is not None:
            #     l = self.weights[i] * l
            losses.append(l)
        return torch.stack(losses).mean()


# ----------------------------------------------------------------------------------------------------------------------
def loop_for_cnn_training(
    base_cnn_model,
    train_dataloader,
    val_dataloader,
    lr,
    max_epochs,
    patience,
    output_dir,
    verbose=True,
):
    print("Model device:", next(base_cnn_model.parameters()).device)

    if verbose:
        print("\n\n----------CNN TRAINING----------\n")

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Output folder to save trained model: {output_dir}")

    loss_criterion = MultiOutputMSELoss()
    optimizer = torch.optim.Adam(base_cnn_model.parameters(), lr=lr)

    # 🔑 AMP scaler
    # scaler = GradScaler('cuda')

    best_model_state_ = None
    best_val_loss = float("inf")
    early_stop_ = False
    epochs_no_improve = 0

    history_path = os.path.join(output_dir, "training_history.pt")

    if os.path.exists(history_path):
        history = torch.load(history_path)
    else:
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
        }


    for epoch in range(max_epochs):
        base_cnn_model.train()
        running_loss = 0.0
        num_batches = 0

        if verbose:
            print(f"\nEpoch {epoch + 1}\n")

        for batch_idx, batch in enumerate(train_dataloader):

            # print("CPU RAM:", psutil.virtual_memory().used / 1e9, "GB")
            # print("GPU mem:", torch.cuda.memory_allocated() / 1e9, "GB")

            if batch is None:
                continue

            # if epoch == 0 and batch_idx == 0:
            #     print("Input dtype:", x_train[0].dtype)
            #     print("Weight dtype:", next(base_cnn_model.parameters()).dtype)
            #     print("Output dtype:", outputs_[0].dtype)

            shot_id, _, x_train, y_train = batch  # (shot_id, window_id, x_train, y_train)
            # print(np.unique(shot_id))
            actual_batch_size = y_train[0].shape[0]
            # if verbose:
                # print(f"Batch {batch_idx} size is {actual_batch_size}")
            # x_train = [arr.to(torch.float32).to(device) for arr in x_train]
            # y_train = [arr.to(torch.float32).to(device) for arr in y_train]

            x_train = [arr.to(torch.float32).to(device).requires_grad_(True) for arr in x_train]
            # print("Checkpoint active, grad:", x_train[0].requires_grad)
            y_train = [arr.to(torch.float32).to(device) for arr in y_train]

            optimizer.zero_grad(set_to_none=True)

            # # 🔑 AMP forward + loss
            # with autocast('cuda'):
            #     outputs_ = base_cnn_model(*x_train)
            #     loss_ = loss_criterion(outputs_, y_train)

            # # 🔑 AMP backward
            # scaler.scale(loss_).backward()
            # scaler.step(optimizer)
            # scaler.update()

            outputs_ = base_cnn_model(*x_train)
            loss_ = loss_criterion(outputs_, y_train)
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

            # if verbose:
            
            #     print(f"Batch loss: {loss_}")

            running_loss += loss_.item() * actual_batch_size
            num_batches += actual_batch_size

        avg_loss = running_loss / num_batches

        if verbose:
            print(f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.4f}")

        # Validation phase & Early stopping check

        base_cnn_model.eval()
        val_running_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):

                if batch is None:
                    continue

                _, _, x_val, y_val = batch  # (shot_id, window_id, x_val, y_val)

                actual_batch_size = y_val[0].shape[0]
                x_val = [arr.to(torch.float32).to(device) for arr in x_val]
                y_val = [arr.to(torch.float32).to(device) for arr in y_val]

                # # 🔑 AMP also in validation
                # with autocast():
                #     val_outputs = base_cnn_model(*x_val)
                #     val_loss = loss_criterion(val_outputs, y_val)

                val_outputs = base_cnn_model(*x_val)
                val_loss = loss_criterion(val_outputs, y_val)
                val_running_loss += val_loss.item() * actual_batch_size
                val_batches += actual_batch_size

        avg_val_loss = val_running_loss / val_batches
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(avg_val_loss)

        torch.save(history, history_path)

        if verbose:
            print(
                f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state_ = base_cnn_model.state_dict()

            # Save best model state
            torch.save(best_model_state_, output_dir + "best_model.pt")

        else:
            epochs_no_improve += 1
            if verbose:
                print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                early_stop_ = True
                if verbose:
                    print("Early stopping triggered.")
                break

    return best_model_state_, early_stop_


# ----------------------------------------------------------------------------------------------------------------------
# CNN EVALUATION
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def NEW_cnn_unstd_evaluation_per_shot(
    test_dataloader,
    config_task,
    cnn_model,
    output_dir,
    window_metrics
    # device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate CNN per shot/window and save incremental RMSEs to CSV.
    """

    print("in cnn_unstd_evaluation_per_shot")

    best_model_path = output_dir + "best_model.pt"
    # csv_path = output_dir + f"{config_task['task_name']}_evaluation_per_window_NEW.csv"

    # remove old file if present
    # if os.path.exists(csv_path):
    #     os.remove(csv_path)

    # Load best model
    cnn_model.load_state_dict(torch.load(best_model_path, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

    feature_names = config_task["sources_and_signals"].get("output_name", [])
    
    dict_task_metadata = get_task_metadata(
        config_task,
        verbose=False
    )

    # Initialize CSV if it doesn’t exist
    # if not os.path.exists(csv_path):
    #     pd.DataFrame(
    #         # columns=["shot_id", "window_id", "feature_name", "global_mean", "global_std", "RMSE", "MSE", "MAE"]
    #         columns=["shot_id", "window_id", "feature_name", "norm", "RMSE", "MSE", "MAE"]
    #     ).to_csv(csv_path, index=False)

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

            # batch_rows = []

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

                window_metrics.compute_and_append(np.float128(unstd_y_t), np.float128(unstd_y_p), shot_id, window_id, f"{feature_name[0]}-{feature_name[1]}")

    print(f"💅🏼 UNSTD Evaluation done. RMSEs and MSEs saved (incrementally).")


# ----------------------------------------------------------------------------------------------------------------------
def cnn_sanity_vizu_per_shot(
    test_dataloader,
    config_task,
    cnn_model,
    output_dir,
    n_shot_to_plot = 3
    # device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Plot some shots CNN per shot/window and save incremental RMSEs to CSV.
    """

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
            
            if counter>=n_shot_to_plot:
                break

            if batch is None:
                continue

            counter +=1

            shot_id, window_id, x_test, y_test = batch

            plot_dir = output_dir + f"sanity_vizu/shot_{shot_id[0]}/"
            os.makedirs(plot_dir, exist_ok=True)

            # Move inputs and labels to device
            x_test = [arr.to(torch.float32).to(device) for arr in x_test]
            y_test = [arr.to(torch.float32).to(device) for arr in y_test]

            # Model prediction
            y_pred = cnn_model(*x_test)

            # Make sure y_pred is list-like
            if not isinstance(y_pred, (list, tuple)):
                y_pred = [y_pred]

            batch_rows = []

            # === Compute RMSEs per feature ===
            for i, feature_name in enumerate(feature_names):

                print(f"{feature_name[0]}-{feature_name[1]}")

                y_t = (
                    y_test[i]
                    .detach()
                    .cpu()
                    .squeeze(1)
                    # .reshape(len(shot_id), -1)
                    .numpy()
                )
                y_p = (
                    y_pred[i]
                    .detach()
                    .cpu()
                    .squeeze(1)
                    # .reshape(len(shot_id), -1)
                    .numpy()
                )

                mean = dict_task_metadata['output'][f"{feature_name[0]}-{feature_name[1]}"]['mean']
                std = dict_task_metadata['output'][f"{feature_name[0]}-{feature_name[1]}"]['std']

                unstd_y_t = y_t*std + mean
                unstd_y_p = y_p*std + mean 

                print('unstd_y_t.shape', unstd_y_t.shape)
                print('unstd_y_t.ndim', unstd_y_t.ndim)

                if unstd_y_t.ndim==1:

                    plt.figure(figsize=(10,5))

                    plt.plot(unstd_y_t, label=f"True")
                    plt.plot(unstd_y_p, '--', label=f"Pred")

                    plt.title("True vs Predicted")
                    plt.xlabel("Sample")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)

                elif unstd_y_t.ndim==2 and unstd_y_t.shape[1]==2:

                    print('hey')

                    plt.figure(figsize=(10,5))

                    plt.plot(unstd_y_t[:, 0], label=f"True")
                    plt.plot(unstd_y_p[:, 0], '--', label=f"Pred")
                
                    plt.plot(unstd_y_t[:, 1], label=f"True")
                    plt.plot(unstd_y_p[:, 1], '--', label=f"Pred")

                    plt.title("True vs Predicted")
                    plt.xlabel("Sample")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)
                
                elif unstd_y_t.ndim == 2 and unstd_y_t.shape[1]!=2:

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
                
                elif (unstd_y_t.ndim==3 and unstd_y_t.shape[2] == 2):
                    
                    n_dim = unstd_y_t.shape[-1]
                    print(unstd_y_t.shape)

                    plt.figure(figsize=(10,5))

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
                
                elif unstd_y_t.ndim==4 :

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

                    plt.figure(figsize=(10,5))

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
                plt.savefig(plot_dir + f"prediction_plot_{feature_name[0]}-{feature_name[1]}.png", dpi=300, bbox_inches="tight")
                plt.close()  # <- this frees memory

    print(f"Visualization saved. Go check in: {output_dir}")

