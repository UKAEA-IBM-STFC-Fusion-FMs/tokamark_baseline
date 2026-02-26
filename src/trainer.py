import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from line_profiler import profile

# Set device
from MAST_benchmark.tools.utils import get_device
device = get_device()
# print(f"Using device: {device}\n")

# ----------------------------------------------------------------------------------------------------------------------
# COLLATE FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
@profile
def cnn_collate_fn(batch, verbose=False):

    # proc = psutil.Process(os.getpid())
    # mem = proc.memory_info().rss / (1024**2)
    # print(f"[Worker PID={proc.pid}] Memory={mem:.2f} MB")

    # Flatten the batch of lists into a single list
    # full_flattened_batch = [
    #     (item["shot_id"], item["window_index"], 
    #     [np.nan_to_num(np.array(x), nan=0.0) for x in item["x"]],
    #     [np.nan_to_num(np.array(y), nan=0.0) for y in item["y"]])
    #     for sublist in batch
    #     for item in sublist
    # ]

    full_flattened_batch = [
        (item["shot_id"], item["window_index"], 
        [np.nan_to_num(np.array(x), nan=0.0) for x in item["x"]],
        [np.nan_to_num(np.array(y), nan=0.0) for y in item["y"]])
        for item in batch
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
            # print(
            #     f"[Output {i}] "
            #     f"target_abs_mean={yt.abs().mean().item():.6f} | "
            #     f"pred_abs_mean={yp.abs().mean().item():.6f}"
            # )                    
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
# @profile
# def loop_for_cnn_training(
#     base_cnn_model,
#     train_dataloader,
#     val_dataloader,
#     lr,
#     max_epochs,
#     patience,
#     output_dir,
#     verbose=True,
# ):
#     print("Model device:", next(base_cnn_model.parameters()).device)

#     if verbose:
#         print("\n\n----------CNN TRAINING----------\n")

#     os.makedirs(output_dir, exist_ok=True)

#     if verbose:
#         print(f"Output folder to save trained model: {output_dir}")

#     loss_criterion = MultiOutputMSELoss()
#     optimizer = torch.optim.Adam(base_cnn_model.parameters(), 
#                                  lr=lr,
#                                  weight_decay=1e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

#     # 🔑 AMP scaler
#     # scaler = GradScaler('cuda')

#     best_model_state_ = None
#     best_val_loss = float("inf")
#     early_stop_ = False
#     epochs_no_improve = 0

#     history_path = os.path.join(output_dir, "training_history.pt")

#     if os.path.exists(history_path):
#         history = torch.load(history_path)
#     else:
#         history = {
#             "epoch": [],
#             "train_loss": [],
#             "val_loss": [],
#         }


#     # single_batch = next(iter(train_dataloader))

#     for epoch in range(max_epochs):
#         base_cnn_model.train()
#         running_loss = 0.0
#         num_batches = 0

#         if verbose:
#             print(f"\nEpoch {epoch + 1}\n")

#         for batch_idx, batch in enumerate(train_dataloader):
#         # for batch_idx in range(2000):   # repeat same batch
#         #     print(batch_idx)
#         #     batch = single_batch

#             # print("CPU RAM:", psutil.virtual_memory().used / 1e9, "GB")
#             # print("GPU mem:", torch.cuda.memory_allocated() / 1e9, "GB")

#             if batch is None:
#                 continue

#             shot_id, _, x_train, y_train = batch  # (shot_id, window_id, x_train, y_train)
#             actual_batch_size = y_train[0].shape[0]
#             # if verbose:
#             #     print(f"Batch {batch_idx} size is {actual_batch_size}")


#             # x_train = [arr.to(torch.float32).to(device).requires_grad_(True) for arr in x_train]
#             x_train = [arr.to(torch.float32).to(device) for arr in x_train]

#             # print("Checkpoint active, grad:", x_train[0].requires_grad)
#             y_train = [arr.to(torch.float32).to(device) for arr in y_train]

#             optimizer.zero_grad(set_to_none=True)

#             # # 🔑 AMP forward + loss
#             # with autocast('cuda'):
#             #     outputs_ = base_cnn_model(*x_train)
#             #     loss_ = loss_criterion(outputs_, y_train)

#             # # 🔑 AMP backward
#             # scaler.scale(loss_).backward()
#             # scaler.step(optimizer)
#             # scaler.update()

#             outputs_ = base_cnn_model(*x_train)
#             loss_ = loss_criterion(outputs_, y_train)

#             # optimizer.zero_grad()
#             loss_.backward()
#             optimizer.step()

#             # if verbose:
#             #     print(f"Batch (size {len(shot_id)}), loss: {loss_}")
#             #     print(f"Batch size: {len(shot_id)}")


#             running_loss += loss_.item() * actual_batch_size
#             num_batches += actual_batch_size

#         # Step the scheduler at the end of each epoch
        
#         scheduler.step()
#         avg_loss = running_loss / num_batches

#         if verbose:
#             print(f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.4f}")

#         # Validation phase & Early stopping check

#         base_cnn_model.eval()
#         val_running_loss = 0.0
#         val_batches = 0

#         with torch.no_grad():
#             for batch_idx, batch in enumerate(val_dataloader):

#                 if batch is None:
#                     continue

#                 _, _, x_val, y_val = batch  # (shot_id, window_id, x_val, y_val)

#                 actual_batch_size = y_val[0].shape[0]
#                 x_val = [arr.to(torch.float32).to(device) for arr in x_val]
#                 y_val = [arr.to(torch.float32).to(device) for arr in y_val]

#                 # # 🔑 AMP also in validation
#                 # with autocast():
#                 #     val_outputs = base_cnn_model(*x_val)
#                 #     val_loss = loss_criterion(val_outputs, y_val)

#                 val_outputs = base_cnn_model(*x_val)
#                 val_loss = loss_criterion(val_outputs, y_val)
#                 val_running_loss += val_loss.item() * actual_batch_size
#                 val_batches += actual_batch_size

#         avg_val_loss = val_running_loss / val_batches
#         history["epoch"].append(epoch + 1)
#         history["train_loss"].append(avg_loss)
#         history["val_loss"].append(avg_val_loss)

#         torch.save(history, history_path)

#         if verbose:
#             print(
#                 f"Epoch [{epoch + 1}/{max_epochs}], Average Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
#             )

#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             epochs_no_improve = 0
#             best_model_state_ = base_cnn_model.state_dict()

#             # Save best model state
#             torch.save(best_model_state_, output_dir + "best_model.pt")

#         else:
#             epochs_no_improve += 1
#             if verbose:
#                 print(f"No improvement for {epochs_no_improve} epochs.")
#             if epochs_no_improve >= patience:
#                 early_stop_ = True
#                 if verbose:
#                     print("Early stopping triggered.")
#                 break

#     return best_model_state_, early_stop_


def move_batch_to_device(batch, device):
    _, _, x, y = batch
    x = [t.to(device=device, dtype=torch.float32) for t in x]
    y = [t.to(device=device, dtype=torch.float32) for t in y]
    return x, y


def train_step(model, batch, optimizer, criterion, device):
    x, y = move_batch_to_device(batch, device)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(*x)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    return loss.item(), y[0].shape[0]


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        if batch is None:
            continue

        x, y = move_batch_to_device(batch, device)

        outputs = model(*x)
        loss = criterion(outputs, y)

        bs = y[0].shape[0]
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / total_samples



import torch
import os

@profile
def train_loop(
    model,
    train_loader,
    val_loader,
    lr,
    max_epochs,
    patience,
    output_dir,
    verbose=True,
):

    os.makedirs(output_dir, exist_ok=True)
    history_file = os.path.join(output_dir, "training_history.pt")

    if verbose:
        print(f"Output folder to save trained model: {output_dir}")

    # Load previous history if it exists
    if os.path.exists(history_file):
        history = torch.load(history_file)
        if verbose:
            print("Loaded previous training history.")
    else:
        history = {"train_loss": [], "val_loss": []}

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion = MultiOutputMSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()

        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            if batch is None:
                continue

            loss, bs = train_step(
                model,
                batch,
                optimizer,
                criterion,
                device,
            )

            running_loss += loss * bs
            total_samples += bs

        train_loss = running_loss / total_samples
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Append to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        torch.save(history, history_file)  # Save after every epoch

        if verbose:
            print(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    return history



import torch
import os

class BatchStepTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr,
        max_steps,
        patience,
        output_dir,
        device,
        validate_every=100,  # validate every k batches
        verbose=True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.max_steps = max_steps
        self.patience = patience
        self.output_dir = output_dir
        self.device = device
        self.verbose = verbose
        self.validate_every = validate_every

        os.makedirs(output_dir, exist_ok=True)
        self.history_file = os.path.join(output_dir, "training_history.pt")

        # Load previous history if exists
        # if os.path.exists(self.history_file):
        #     self.history = torch.load(self.history_file)
        #     if verbose:
        #         print("Loaded previous training history.")
        # else:
        # Step-level history
        self.history = {
            "step": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=1000
        # )
        self.criterion = MultiOutputMSELoss()

        self.best_val_loss = float("inf")
        self.valid_no_improve = 0
        self.current_step = 0
        self.current_batch_loss = []  # track running train loss
        self.batch_iter = iter(self.train_loader)

    def step_batch(self):
        """Perform a single batch update. Validate every `validate_every` batches."""
        try:
            batch = next(self.batch_iter)
        except StopIteration:
            # End of epoch, reset iterator
            self.batch_iter = iter(self.train_loader)
            batch = next(self.batch_iter)
            self.current_batch_loss = []  # reset running loss per epoch

        if batch is None:
            return True  # skip empty batch

        # ---- Train step ----
        loss, bs = train_step(
            self.model,
            batch,
            self.optimizer,
            self.criterion,
            self.device,
        )
        self.current_batch_loss.append(loss)
        self.current_step += 1

        # # ---- Update scheduler ----
        # self.scheduler.step()

        # ---- Current learning rate ----
        # current_lr = self.optimizer.param_groups[0]["lr"]

        # ---- Average train loss so far in this step/epoch ----
        avg_train_loss = sum(self.current_batch_loss) / len(self.current_batch_loss)

        # ---- Perform validation every k batches ----
        val_loss = None
        if self.current_step % self.validate_every == 0:
            val_loss = validate(self.model, self.val_loader, self.criterion, self.device)

            # ---- Save history for this step ----
            self.history["step"].append(self.current_step)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)  # None if not validated
            # self.history["lr"].append(current_lr)
            torch.save(self.history, self.history_file)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.valid_no_improve = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, "best_model.pt"),
                )
            else:
                self.valid_no_improve += 1
                if self.valid_no_improve >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered.")
                    return False

            # ---- Print info ----
            if self.verbose:
                # msg = f"Step {self.current_step} | train_loss={avg_train_loss:.4f} | lr={current_lr:.6f}"
                msg = f"Step {self.current_step} | train_loss={avg_train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.4f}"
                print(msg)

        # ---- Stop if max steps reached ----
        if self.current_step >= self.max_steps:
            if self.verbose:
                print(f"Max steps {self.max_steps} reached.")
            return False

        return True
