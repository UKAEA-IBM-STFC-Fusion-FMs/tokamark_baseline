
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchinfo import summary
# import torch.nn.functional as F
# import numpy as np

from src.model_transform import _make_dummy_outputs  # _resample
from src.conv_encoders_decoders import Conv1DEncoder, Conv2DEncoder, Conv3DEncoder, Conv1DDecoder, Conv2DDecoder, Conv3DDecoder
from tokamark.tools.utils import get_device

# ----------------------------------------------------------------------------------------------------------------------
# Set device
device = get_device()
# print(f"Using device: {device}\n")

padding = 1
kernel_size = 3
stride = 3
layers_encoder = 3
layers_decoder = 3
bb_factor = 2

# ----------------------------------------------------------------------------------------------------------------------
def create_cnn_architecture(dataloader_, dict_metadata, D=16, verbose=True):
    
    if verbose:
        print("\n\n----------CNN MODEL INITIALIZATION----------\n")

    # ------------------------------------------------------------
    # 1. Extract one valid sample for shape inference
    # ------------------------------------------------------------

    input_shapes = []
    exogenous_shapes = []
    output_shapes = []

    for l, first_window in enumerate(dataloader_.dataset):

        try:
            input_shapes = [arr.shape for arr in first_window["input"]]
            exogenous_shapes = [arr.shape for arr in first_window["exogenous"]]
            output_shapes = [arr.shape for arr in first_window["y"]]

            if verbose:
                print(f"Shot {first_window['shot_id']} used as reference")
                print(f"Input shapes are: {input_shapes}")
                print(f"Actuator future shapes are: {exogenous_shapes}")
                print(f"Output shapes are: {output_shapes}")

            break

        except Exception as e:
            print(f"Skipping sample {l} because not trainable: {e}")
            continue

    # ------------------------------------------------------------
    # 2. Create model (your CNN + Window + LSTM model)
    # ------------------------------------------------------------
    model = MultiConv_MLP(
        input_shapes=input_shapes,
        exogenous_shapes=exogenous_shapes,
        output_shapes=output_shapes,
        dict_metadata=dict_metadata,
        D=D,
    ).to(device)

    # ------------------------------------------------------------
    # 3. Build dummy input sizes for torchinfo
    #    IMPORTANT: we only pass raw tensors BEFORE windowing
    # ------------------------------------------------------------
    input_sizes = []

    for shape in ( input_shapes + exogenous_shapes ):
        input_sizes.append((2,) + shape)

    # ------------------------------------------------------------
    # 4. Model summary
    # ------------------------------------------------------------
    if verbose:
        summary(model, input_size=input_sizes)

    return model

# ======================================================================================================================
class MultiConv_MLP(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(  # NOSONAR - Ignore cognitive complexity
        self,
        input_shapes,
        exogenous_shapes,
        output_shapes,
        dict_metadata,
        D=16
    ):

        super().__init__()

        self.D = D
        self.W_in = input_shapes[0][0]

        self.output_shapes = output_shapes
        y = _make_dummy_outputs(output_shapes, dict_metadata)
        output_latent_shapes = [arr.shape for arr in y]
        self.W_out = output_latent_shapes[0][0]

        # --------------------------------------------------------------------------------------------------------------
        self.input_branches = nn.ModuleList()

        for var_shape in input_shapes:
            if len(var_shape) == 5:  # e.g., (2, T, 15, 17) images evolving in time
                branch = Conv3DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 4:  # e.g., (1, T, 15) profiles evolving in time
                branch = Conv2DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 3:  # e.g., (7, T, ) time series evolving in time
                branch = Conv1DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.input_branches.append(branch)

        # --------------------------------------------------------------------------------------------------------------
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.D * bb_factor * len(self.input_branches) * self.W_in, 2 * self.D * bb_factor),
            nn.ReLU(),
            nn.Linear(2 * self.D * bb_factor, self.D * bb_factor),
        )

        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.D * bb_factor * (1 + len(exogenous_shapes) * self.W_out ) , 2 * self.D * bb_factor),
            nn.ReLU(),
            nn.Linear(2 * self.D * bb_factor, self.D * bb_factor),
        )

        # --------------------------------------------------------------------------------------------------------------
        self.exogenous_branches = nn.ModuleList()

        for var_shape in exogenous_shapes:
            if len(var_shape) == 5:  # e.g., (2, T, 15, 17) images evolving in time
                branch = Conv3DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 4:  # e.g., (1, T, 15) profiles evolving in time
                branch = Conv2DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 3:  # e.g., (7, T, ) time series evolving in time
                branch = Conv1DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding, bb_factor)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.exogenous_branches.append(branch)

        # --------------------------------------------------------------------------------------------------------------
        self.output_branches = nn.ModuleList()

        for var_shape in output_latent_shapes:

            if len(var_shape) == 5:
                branch = Conv3DDecoder(var_shape[1:], D,  layers_decoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 4:
                branch = Conv2DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding, bb_factor)
            elif len(var_shape) == 3:
                branch = Conv1DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding, bb_factor)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.output_branches.append(branch)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _run_cnn_encoder(branch, x):
        B, W = x.shape[:2]  # batch, num_windows
        # merge batch and window dims
        x = x.view(B * W, *x.shape[2:])
        # pass through CNN
        out = branch(x)
        # restore dimensions
        out = out.view(B, W, -1)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _run_cnn_decoder(branch, goal_shape, x):

        B, W = x.shape[:2]

        # merge batch + windows for CNN
        x = x.reshape(B * W, *x.shape[2:])

        out = branch(x)

        # restore batch structure
        out = out.reshape(B, W * out.shape[2], *out.shape[3:])

        target_len = goal_shape[0]

        out = out[:, -target_len:]

        return out

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, *args):

        
        # -----------------------------
        # CNN ENCODER
        # -----------------------------

        n_in = len(self.input_branches)
        n_exo = len(self.exogenous_branches)

        inputs = args[:n_in]
        exogenous = args[n_in:n_in + n_exo]

        # -----------------------------
        # input encoding
        # -----------------------------
        input_branch_outputs = []
        for branch, x in zip(self.input_branches, inputs):
            out = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
            input_branch_outputs.append(out)

        input_seq = torch.cat(input_branch_outputs, dim=2)

        # -----------------------------
        # exogenous encoding
        # -----------------------------
        exo_branch_outputs = []

        for branch, x in zip(self.exogenous_branches, exogenous):
            out = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
            exo_branch_outputs.append(out)

        if len(exo_branch_outputs) > 0:
            exo_seq = torch.cat(exo_branch_outputs, dim=2)
        else:
            exo_seq = None

        # -----------------------------
        # BACKBONE (MLP VERSION)
        # -----------------------------

        B, W_in, D_in = input_seq.shape

        # -----------------------------
        # ENCODER MLP
        # -----------------------------
        enc_in = input_seq.reshape(B, W_in * D_in)

        context = self.encoder_mlp(enc_in)   # (B, D)

        # -----------------------------
        # DECODER INPUT PREP
        # -----------------------------
        if exo_seq is not None:
            B, W_out, D_exo = exo_seq.shape
            exo = exo_seq.reshape(B, W_out * D_exo)
            decoder_in = torch.cat([context, exo], dim=1) #SAHPE HERE IS D (1 + W_out_D_exo)
        else:
            decoder_in = context

        dec_flat = self.decoder_mlp(decoder_in)  # (B, D)

        # -----------------------------
        # EXPAND TO TIME DIM
        # -----------------------------
        dec_out = dec_flat.unsqueeze(1).repeat(1, self.W_out, 1)

        # -----------------------------
        # CNN DECODERS
        # -----------------------------

        # -----------------------------
        # output heads
        # -----------------------------
        outputs = []
        for branch, goal_shape in zip(self.output_branches, self.output_shapes):
            out = checkpoint(self._run_cnn_decoder, branch, goal_shape, dec_out, use_reentrant=False)
            outputs.append(out)

        return outputs