from globals import REPO_ROOT
from torchinfo import summary

# Set device
from MAST_benchmark.tools.utils import get_device

device = get_device()
# print(f"Using device: {device}\n")

import os
import psutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from time_cnn_model import MultiBranchTimeCNNModel

# ----------------------------------------------------------------------------------------------------------------------
# COLLATE FUNCTION
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def cnn_training_collate_fn(batch, verbose=False):
    # print(f"Collating batch of size {len(batch)}")

    proc = psutil.Process(os.getpid())
    mem = proc.memory_info().rss / (1024**2)
    # print(f"[Worker PID={proc.pid}] Memory={mem:.2f} MB")

    # Flatten the batch of lists into a single list
    # print(batch)
    flattened_batch = [
        (item["shot_id"], item["window_index"], item["x"], item["y"])
        for sublist in batch
        for item in sublist
        if not (
            any(np.isnan(np.array(x)).any() for x in item["x"])
            or any(np.isnan(np.array(y)).any() for y in item["y"])
        )
    ]

    if verbose:
        print(
            f"Number of samples from batch = {len(batch)} shots is N = {len(flattened_batch)}"
        )
        if len(flattened_batch) == 0:
            print("batch is None")

    return default_collate(flattened_batch) if (len(flattened_batch) > 0) else None


# ----------------------------------------------------------------------------------------------------------------------
# CNN TRAINING
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def create_cnn_architecture(dataloader_, D, verbose=False):
    if verbose:
        print("\n\n----------MODEL INITIALIZATION----------\n")

    for l in range(len(dataloader_.dataset)):
        try:
            # Get the generator from __getitem__
            windows_gen = dataloader_.dataset[l]  # this is now a generator
            first_window = next(windows_gen)  # get the first yielded window
            input_shapes = [arr.shape for arr in first_window["x"]]
            output_shape = [arr.shape for arr in first_window["y"]]

            # input_shapes = [arr.shape for arr in dataloader_.dataset[l][0]['x']]
            # output_shape = [arr.shape for arr in dataloader_.dataset[l][0]['y']]

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

    cnn_model = MultiBranchTimeCNNModel(input_shapes, output_shape, D).to(device)

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

    best_model_state_ = None
    best_val_loss = float("inf")
    early_stop_ = False
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        base_cnn_model.train()
        running_loss = 0.0
        num_batches = 0

        if verbose:
            print(f"\nEpoch {epoch + 1}\n")

        for batch_idx, batch in enumerate(train_dataloader):
            if batch is None:
                continue

            _, _, x_train, y_train = batch  # (shot_id, window_id, x_train, y_train)
            actual_batch_size = y_train[0].shape[0]
            # if verbose:
                # print(f"Batch {batch_idx} size is {actual_batch_size}")
            x_train = [arr.to(torch.float32).to(device) for arr in x_train]
            y_train = [arr.to(torch.float32).to(device) for arr in y_train]

            outputs_ = base_cnn_model(*x_train)
            loss_ = loss_criterion(outputs_, y_train)
            if verbose:
                print(f"Batch loss: {loss_}")

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

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

                val_outputs = base_cnn_model(*x_val)
                val_loss = loss_criterion(val_outputs, y_val)
                val_running_loss += val_loss.item() * actual_batch_size
                val_batches += actual_batch_size

        avg_val_loss = val_running_loss / val_batches

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
def cnn_evaluation_per_shot(
    test_dataloader,
    config_task,
    cnn_model,
    config_cnn,
    # device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate CNN per shot/window and save incremental RMSEs to CSV.
    """

    # === Setup paths ===
    output_dir = (
        REPO_ROOT
        + config_cnn["paths"]["data_output_directory"]
        + config_task["task_name"]
    )
    best_model_path = output_dir + "/best_model.pt"
    csv_path = output_dir + "/rmse_and_mse_per_sample.csv"

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Load best model
    cnn_model.load_state_dict(torch.load(best_model_path, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

    feature_names = config_task["sources_and_signals"].get("output_name", [])

    # Initialize CSV if it doesn’t exist
    if not os.path.exists(csv_path):
        pd.DataFrame(
            columns=["shot_id", "window_id", "feature_name", "RMSE", "MSE"]
        ).to_csv(csv_path, index=False)

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

            batch_rows = []

            # === Compute RMSEs per feature ===
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

                rmse_per_sample = np.sqrt(np.mean((y_t - y_p) ** 2, axis=1))
                mse_per_sample = np.mean((y_t - y_p) ** 2, axis=1)

                for sid, wid, rmse_val, mse_val in zip(
                    shot_id, window_id, rmse_per_sample, mse_per_sample
                ):
                    batch_rows.append(
                        {
                            "shot_id": sid.item() if torch.is_tensor(sid) else sid,
                            "window_id": wid.item() if torch.is_tensor(wid) else wid,
                            "feature_name": f"{feature_name[0]}-{feature_name[1]}",
                            "RMSE": rmse_val,
                            "MSE": mse_val,
                        }
                    )

            # === Append to CSV ===
            df_batch = pd.DataFrame(batch_rows)
            df_batch.to_csv(csv_path, mode="a", header=False, index=False)

    print(f"✅ Evaluation done. RMSEs and MSEs saved (incrementally) to: {csv_path}")


# ----------------------------------------------------------------------------------------------------------------------
def cnn_save_traces_per_shot(
    test_dataloader,
    config_task,
    cnn_model,
    config_cnn,
    n_traces=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Save per-feature, per-shot traces from CNN predictions.

    Folder structure:
      output_dir/
        ├── <feature_name>/
        │     ├── <shot_id>/trace.npz

    NPZ content:
      - true:        (N_windows, 1, *spatial_shape)
      - pred:        (N_windows, 1, *spatial_shape)
      - window_idx:  (N_windows,)
    """

    # === Paths ===
    output_dir = (
        REPO_ROOT
        + config_cnn["paths"]["data_output_directory"]
        + config_task["task_name"]
    )
    best_model_path = os.path.join(output_dir, "best_model.pt")

    output_root_traces = output_dir + "/traces/"
    os.makedirs(output_root_traces, exist_ok=True)

    # === Model setup ===
    cnn_model.load_state_dict(torch.load(best_model_path, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

    feature_names = config_task["sources_and_signals"].get("output", [])

    # Container for grouping all windows of each shot per feature
    traces = {}

    saved_traces = 0

    # === Save one NPZ per shot per feature ===
    with torch.no_grad():
        for i, (shot_id, window_id, x_test, y_test) in enumerate(test_dataloader):
            if saved_traces >= n_traces:
                break

            # Move data to device
            x_test = [arr.to(torch.float32).to(device) for arr in x_test]
            y_test = [arr.to(torch.float32).to(device) for arr in y_test]

            y_pred = cnn_model(*x_test)
            if not isinstance(y_pred, (list, tuple)):
                y_pred = [y_pred]

            # Convert IDs
            shot_ids_np = (
                shot_id.detach().cpu().numpy()
                if torch.is_tensor(shot_id)
                else np.array(shot_id)
            )
            window_ids_np = (
                window_id.detach().cpu().numpy()
                if torch.is_tensor(window_id)
                else np.array(window_id)
            )

            # Process each predicted feature
            for i, feature_name in enumerate(feature_names):
                print(feature_name)
                y_t = y_test[i].detach().cpu().numpy()  # (N, 1, ...)
                y_p = y_pred[i].detach().cpu().numpy()

                for j, sid in enumerate(shot_ids_np):
                    wid = window_ids_np[j]
                    key = (
                        (feature_name[0], feature_name[1], sid)
                        if isinstance(feature_name, (list, tuple))
                        else (feature_name, sid)
                    )

                    if key not in traces:
                        traces[key] = {"true": [], "pred": [], "window_idx": []}

                    traces[key]["true"].append(y_t[j : j + 1])
                    traces[key]["pred"].append(y_p[j : j + 1])
                    traces[key]["window_idx"].append(wid)

            saved_traces += np.unique(shot_ids_np).size

    for key, data in traces.items():
        if len(key) == 3:
            f_src, f_sig, sid = key
            feature_dir_name = f"{f_src}-{f_sig}"
        else:
            feature_dir_name, sid = key

        feature_dir = os.path.join(output_root_traces, feature_dir_name)
        shot_dir = os.path.join(feature_dir, str(sid))
        os.makedirs(shot_dir, exist_ok=True)

        true_arr = np.concatenate(data["true"], axis=0)
        pred_arr = np.concatenate(data["pred"], axis=0)
        window_arr = np.array(data["window_idx"])

        np.savez(
            os.path.join(shot_dir, "trace.npz"),
            true=true_arr,
            pred=pred_arr,
            window_idx=window_arr,
        )

    print(
        f"✅ Saved {saved_traces} full-shot traces under {output_root_traces}<feature_name>/<shot_id>/trace.npz"
    )
