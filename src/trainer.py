import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

# Set device
from MAST_benchmark.tools.utils import get_device
device = get_device()
# print(f"Using device: {device}\n")

# ----------------------------------------------------------------------------------------------------------------------
# COLLATE FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def cnn_collate_fn(batch, verbose=False):

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
# CNN LOSS
# ----------------------------------------------------------------------------------------------------------------------

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
# TRAINING LOOP
# ----------------------------------------------------------------------------------------------------------------------

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

            shot_id, _, x_train, y_train = batch  # (shot_id, window_id, x_train, y_train)
            actual_batch_size = y_train[0].shape[0]
            # if verbose:
                # print(f"Batch {batch_idx} size is {actual_batch_size}")


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
