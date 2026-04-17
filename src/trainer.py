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
def cnn_collate_fn(batch, verbose=False):

    full_flattened_batch = [
        (
            item["shot_id"],
            item["window_index"],
            [np.nan_to_num(np.array(x), nan=0.0) for x in item["x"]],
            [np.nan_to_num(np.array(y), nan=0.0) for y in item["y"]]
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

# ======================================================================================================================
class MultiOutputMSELoss(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, y_preds, y_trues):

        if len(y_preds) != len(y_trues):
            raise ValueError("Mismatch in number of outputs.")

        losses = []
        for i, (yp, yt) in enumerate(zip(y_preds, y_trues)):                 

            if yp.shape != yt.shape:
                raise ValueError(f"Shape mismatch at output {i}: {yp.shape} vs {yt.shape}")

            l_ = F.mse_loss(yp, yt, reduction=self.reduction)
            losses.append(l_)
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
    x, y = move_batch_to_device(batch, device)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(*x)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    return loss.item(), y[0].shape[0]


# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
def train_loop(  # NOSONAR - Ignore cognitive complexity
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
