import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

# Set device
from tokamark.tools.utils import get_device


# ----------------------------------------------------------------------------------------------------------------------

device = get_device()
# print(f"Using device: {device}\n")


# ----------------------------------------------------------------------------------------------------------------------
# COLLATE FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def model_collate_fn(batch, verbose=False):

    # print('in collate model')
    full_flattened_batch = [
        (
            item["shot_id"],
            item["window_index"],
            [np.nan_to_num(np.array(x), nan=0.0) for x in item["input"] + item["exogenous"]],
            item["y"]
            # [np.nan_to_num(np.array(y), nan=0.0) for y in item["y"]]
        )
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

class MultiOutputMSELoss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_preds, y_trues):

        if len(y_preds) != len(y_trues):
            raise ValueError("Mismatch in number of outputs.")

        losses = []
        for i, (yp, yt) in enumerate(zip(y_preds, y_trues)):

            if yp.shape != yt.shape:
                raise ValueError(f"Shape mismatch at output {i}: {yp.shape} vs {yt.shape}")

            mask = ~torch.isnan(yt)

            if mask.sum() == 0:
                # No valid targets → skip or append 0 (depends on your preference)
                continue

            diff = yp[mask] - yt[mask]
            l_ = (diff ** 2)

            if self.reduction == "mean":
                l_ = l_.mean()
            elif self.reduction == "sum":
                l_ = l_.sum()

            losses.append(l_)

        if len(losses) == 0:
            return torch.tensor(0.0, device=y_preds[0].device)

        # print(torch.stack(losses).mean())
        return torch.stack(losses).mean()


# ----------------------------------------------------------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
def move_batch_to_device(batch, device):
    _, _, x, y = batch
    x = [t.to(device=device, dtype=torch.float32) for t in x]
    y = [t.to(device=device, dtype=torch.float32) for t in y]
    return x, y


# ----------------------------------------------------------------------------------------------------------------------
def train_step(model, batch, optimizer, criterion, device):
    model.train()  # Ensure model is in training mode
    
    x, y = move_batch_to_device(batch, device)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(*x)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    return loss.item(), y[0].shape[0]


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, criterion, device, count_stats=False, return_stats=False):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    
    # Track unique shots if requested
    unique_shots = set() if count_stats else None

    for batch in loader:
        if batch is None:
            continue
        
        # Count unique shots if requested
        if count_stats and unique_shots is not None:
            shot_ids, _, _, _ = batch
            unique_shots.update(shot_ids.cpu().numpy())

        x, y = move_batch_to_device(batch, device)

        outputs = model(*x)
        loss = criterion(outputs, y)

        bs = y[0].shape[0]
        total_loss += loss.item() * bs
        total_samples += bs
    
    avg_loss = total_loss / total_samples
    
    # Return statistics if requested
    if return_stats:
        if count_stats and unique_shots is not None:
            stats = {
                "samples": total_samples,
                "unique_shots": len(unique_shots)
            }
        else:
            stats = None
        return avg_loss, stats
    
    return avg_loss



# ======================================================================================================================
class BatchStepTrainer:

    # ------------------------------------------------------------------------------------------------------------------
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

        self.history = {
            "step": [],
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.criterion = MultiOutputMSELoss()

        self.best_val_loss = float("inf")
        self.valid_no_improve = 0
        self.current_step = 0
        self.current_batch_loss = []  # track running train loss
        self.batch_iter = iter(self.train_loader)
        
        # Track statistics for first epoch
        self.first_epoch_stats_printed = False
        self.train_unique_shots = set()
        self.train_sample_count = 0
        self.epoch_count = 0

    # ------------------------------------------------------------------------------------------------------------------
    def step_batch(  # NOSONAR - Ignore cognitive complexity
            self
    ) -> bool:
        """Perform a single batch update. Validate every `validate_every` batches."""
        try:
            batch = next(self.batch_iter)
        except StopIteration:
            # End of epoch, reset iterator
            self.batch_iter = iter(self.train_loader)
            batch = next(self.batch_iter)
            self.current_batch_loss = []  # reset running loss per epoch
            self.epoch_count += 1
            
            # Save and print first epoch statistics when epoch 0 ends
            if self.epoch_count == 1 and not self.first_epoch_stats_printed:
                # Get validation stats - returns tuple when return_stats=True
                result = validate(
                    self.model, self.val_loader, self.criterion,
                    self.device, count_stats=True, return_stats=True
                )
                # Unpack tuple
                if isinstance(result, tuple):
                    val_loss_temp, val_stats = result
                else:
                    val_loss_temp = result
                    val_stats = None
                
                # Create stats dictionary
                first_epoch_stats = {
                    "train_samples": self.train_sample_count,
                    "train_unique_shots": len(self.train_unique_shots),
                    "val_samples": val_stats["samples"] if val_stats else 0,
                    "val_unique_shots": val_stats["unique_shots"] if val_stats else 0,
                }
                
                # Save to file
                stats_file = os.path.join(self.output_dir, "first_epoch_stats.pt")
                torch.save(first_epoch_stats, stats_file)
                
                # Print statistics
                print(f"\n{'='*60}")
                print(f"FIRST EPOCH STATISTICS:")
                print(f"{'='*60}")
                print(f"Train - Total samples: {self.train_sample_count}")
                print(f"Train - Unique shots: {len(self.train_unique_shots)}")
                print(f"Val - Total samples: {first_epoch_stats['val_samples']}")
                print(f"Val - Unique shots: {first_epoch_stats['val_unique_shots']}")
                print(f"Saved to: {stats_file}")
                print(f"{'='*60}\n")
                self.first_epoch_stats_printed = True

        if batch is None:
            return True  # skip empty batch
        
        # Count unique shots and samples in first epoch
        if self.epoch_count == 0:
            shot_ids, _, _, _ = batch
            self.train_unique_shots.update(shot_ids.cpu().numpy())

        # ---- Train step ----
        loss, bs = train_step(
            self.model,
            batch,
            self.optimizer,
            self.criterion,
            self.device,
        )
        
        # Count samples in first epoch
        if self.epoch_count == 0:
            self.train_sample_count += bs
        
        self.current_batch_loss.append(loss)
        self.current_step += 1

        # ---- Average train loss so far in this step/epoch ----
        avg_train_loss = sum(self.current_batch_loss) / len(self.current_batch_loss)

        # ---- Perform validation every k batches ----
        val_loss = None
        if self.current_step % self.validate_every == 0:
            result = validate(self.model, self.val_loader, self.criterion, self.device, count_stats=False, return_stats=False)
            # Ensure val_loss is a float, not a tuple
            val_loss = result if not isinstance(result, tuple) else result[0]

            # ---- Save history for this step ----
            self.history["step"].append(self.current_step)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)  # None if not validated
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
