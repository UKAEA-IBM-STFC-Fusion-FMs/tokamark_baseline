import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

from tokamark.tools.utils import get_device


# ----------------------------------------------------------------------------------------------------------------------

# Set device
device = get_device()
# print(f"Using device: {device}\n")


def create_lstm_architecture(dataloader_, D, dict_metadata, verbose=True):

    if verbose:
        print("\n\n----------LSTM MODEL INITIALIZATION----------\n")

    # -------------------------------------------------
    # 1. Extract one valid sample (same as CNN version)
    # -------------------------------------------------
    for l, first_window in enumerate(dataloader_.dataset):
        print(l)
        print(first_window['shot_id'])

        try:
            input_shapes = [arr.shape for arr in first_window["x"]]
            output_shapes = [arr.shape for arr in first_window["y"]]

            if verbose:
                print(f"Shot {first_window['shot_id']} used as reference")
                print(f"Input shapes are: {input_shapes}")
                print(f"Output shapes are: {output_shapes}")

            break

        except Exception as e:
            print(f"Skipping sample {l} because not trainable: {e}")
            continue

    # -------------------------------------------------
    # 2. Build LSTM model
    # -------------------------------------------------
    lstm_model = MultiBranchTimeLSTMBaseline(
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        D=D,
        dict_metadata=dict_metadata
    ).to(device)

    # -------------------------------------------------
    # 3. Summary input formatting (IMPORTANT FIX)
    #    LSTM expects (B, T, F), so we adapt shapes
    # -------------------------------------------------
    input_size = []

    for shape in input_shapes:

        if len(shape) == 2:
            # (T, F)
            input_size.append((2, shape[0], shape[1]))
        else:
            # (F,) static → treated as (T=1, F)
            input_size.append((2, 1, shape[0]))

    summary(lstm_model, input_size=input_size)

    return lstm_model


class TemporalAlign(nn.Module):
    def __init__(self, target_T):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(target_T)
        self.target_T = target_T

    def forward(self, x):
        """
        x: (B, T, F)
        """
        x = x.permute(0, 2, 1)      # (B, F, T)
        x = self.pool(x)            # (B, F, T_target)
        x = x.permute(0, 2, 1)      # (B, T_target, F)
        return x

class LearnableTemporalUpsample(nn.Module):
    def __init__(self, channels, target_T):
        super().__init__()

        self.deconv = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.target_T = target_T

    def forward(self, x):
        """
        x: (B, C, T)
        """
        x = self.deconv(x)

        # enforce exact output size per modality
        x = F.interpolate(
            x,
            size=self.target_T,
            mode="linear",
            align_corners=False
        )

        return x

class MultiBranchTimeLSTMBaseline(nn.Module):
    
    # Architecture constants
    LSTM_OUTPUT_DIM = 32
    DECODER_HIDDEN_DIM = 64
    DROPOUT_RATE = 0.2

    def __init__(self, input_shapes, output_shapes, D, dict_metadata):
        super().__init__()

        self.n_inputs = len(input_shapes)
        self.n_outputs = len(output_shapes)
        self.output_shapes = output_shapes  # Store for use in forward pass

        times = set([arr[0] for arr in input_shapes + output_shapes])
        self.T = max(times)

        self.temporal_align = TemporalAlign(target_T=self.T)

        self.temporal_decoder = nn.Linear(self.DECODER_HIDDEN_DIM, self.LSTM_OUTPUT_DIM * self.T)
        self.temporal_channels = self.LSTM_OUTPUT_DIM

        self.temporal_decoders = nn.ModuleList()

        for out_shape in output_shapes:

            if len(out_shape) >= 2:
                T_out = out_shape[0]  # temporal dimension (first dimension)

                self.temporal_decoders.append(
                    LearnableTemporalUpsample(
                        channels=self.temporal_channels,
                        target_T=T_out
                    )
                )

        # ---------------------------
        # INPUT ENCODERS (per branch)
        # ---------------------------
        self.encoders = nn.ModuleList()

        for shape in input_shapes:

            if len(shape) == 3:
                # (C, H, W) - Image input
                C, H, W = shape
                self.encoders.append(
                    nn.Sequential(
                        nn.Flatten(),  # (B, C, H, W) -> (B, C*H*W)
                        nn.Linear(C * H * W, self.DECODER_HIDDEN_DIM),
                        nn.ReLU(),
                        nn.Linear(self.DECODER_HIDDEN_DIM, self.LSTM_OUTPUT_DIM),
                        nn.ReLU()
                    )
                )
            elif len(shape) == 2:
                # (T, F) - Temporal sequence
                self.encoders.append(
                    nn.LSTM(
                        input_size=shape[1],
                        hidden_size=32,
                        batch_first=True
                    )
                )
            else:
                # (F,) - Static features → MLP
                self.encoders.append(
                    nn.Sequential(
                        nn.Linear(shape[0], 32),
                        nn.ReLU()
                    )
                )

        # ---------------------------
        # FUSION LSTM
        # ---------------------------
        self.fusion_lstm = nn.LSTM(
            input_size=32 * self.n_inputs,
            hidden_size=64,
            batch_first=True
        )

        self.latent = nn.Linear(self.DECODER_HIDDEN_DIM, self.DECODER_HIDDEN_DIM)

        # ---------------------------
        # OUTPUT HEADS (multi-output)
        # ---------------------------
        self.decoders = nn.ModuleList()

        for shape in output_shapes:

            if len(shape) == 3:
                # (T, H, W) - Temporal sequence of images
                # Input: (B, T_i, 64) from temporal decoder
                T_out, H, W = shape
                
                self.decoders.append(
                    nn.Sequential(
                        nn.Linear(self.LSTM_OUTPUT_DIM, self.DECODER_HIDDEN_DIM),
                        nn.ReLU(),
                        nn.Dropout(p=self.DROPOUT_RATE),
                        nn.Linear(self.DECODER_HIDDEN_DIM, H * W)
                    )
                )

            elif len(shape) == 2:
                # (T, F) - 2D temporal sequence output
                # Input: (B, T_i, 64) from temporal decoder
                T_out, F_out = shape
                
                self.decoders.append(
                    nn.Sequential(
                        nn.Linear(self.LSTM_OUTPUT_DIM, self.DECODER_HIDDEN_DIM),
                        nn.ReLU(),
                        nn.Dropout(p=self.DROPOUT_RATE),
                        nn.Linear(self.DECODER_HIDDEN_DIM, F_out)
                    )
                )

            else:
                # (F,) - 1D vector output
                # Input: (B, DECODER_HIDDEN_DIM * 2) directly from latent z
                self.decoders.append(
                    nn.Sequential(
                        nn.Linear(self.DECODER_HIDDEN_DIM, self.DECODER_HIDDEN_DIM),
                        nn.ReLU(),
                        nn.Dropout(p=self.DROPOUT_RATE),
                        nn.Linear(self.DECODER_HIDDEN_DIM, shape[0])
                    )
                )

    def forward(self, *inputs):

        encoded = []

        # ---------------------------
        # ENCODE EACH MODALITY
        # ---------------------------
        for x, encoder in zip(inputs, self.encoders):

            if x.dim() == 4:
                # (B, C, H, W) - Image input
                x = encoder(x)  # -> (B, 64)
                out = x.unsqueeze(1).repeat(1, self.T, 1)  # -> (B, T, 64)

            elif x.dim() == 3:
                # Check if it's a static feature (B, 1, F) or temporal sequence (B, T, F)
                if x.shape[1] == 1:
                    # (B, 1, F) - Static features with extra dimension
                    x = x.squeeze(1)  # -> (B, F)
                    x = encoder(x)  # -> (B, 64)
                    out = x.unsqueeze(1).repeat(1, self.T, 1)  # -> (B, T, 64)
                else:
                    # (B, T, F) - Temporal sequence
                    x = self.temporal_align(x)  # Align to target T
                    out, _ = encoder(x)  # -> (B, T, 64)

            else:
                # (B, F) - Static features (2D)
                x = encoder(x)  # -> (B, 64)
                out = x.unsqueeze(1).repeat(1, self.T, 1)  # -> (B, T, 64)

            encoded.append(out)

        # ---------------------------
        # FUSION
        # ---------------------------
        fused = torch.cat(encoded, dim=-1)

        fused, _ = self.fusion_lstm(fused)

        h = fused[:, -1, :]  # last timestep

        z = self.latent(h)

        # ---------------------------
        # MULTI-OUTPUT DECODING (TEMPORAL VERSION)
        # ---------------------------
        outputs = []

        # 1) expand latent into temporal tensor (only for temporal outputs)
        x = self.temporal_decoder(z)                        # (B, C*T)
        x = x.view(-1, self.temporal_channels, self.T)     # (B, C, T)

        # 2) per-output decoding
        temporal_decoder_idx = 0
        for i, out_shape in enumerate(self.output_shapes):

            if len(out_shape) == 3:
                # (T, H, W) - Temporal sequence of images
                T_out, H, W = out_shape
                
                x_i = self.temporal_decoders[temporal_decoder_idx](x)   # (B, C, T_i)
                temporal_decoder_idx += 1
                
                # Permute to (B, T_i, C) for temporal decoding
                x_i = x_i.permute(0, 2, 1)           # (B, T_i, temporal_channels)
                
                # Apply decoder at each timestep
                out = self.decoders[i](x_i)          # (B, T_i, H*W)
                
                # Reshape to image sequence
                out = out.view(-1, T_out, H, W)      # (B, T_i, H, W)

            elif len(out_shape) == 2:
                # (T, F) - 2D temporal sequence output
                x_i = self.temporal_decoders[temporal_decoder_idx](x)   # (B, C, T_i)
                temporal_decoder_idx += 1
                
                # Permute to (B, T_i, C) for temporal decoding
                x_i = x_i.permute(0, 2, 1)           # (B, T_i, temporal_channels)
                
                # Apply decoder at each timestep
                out = self.decoders[i](x_i)          # (B, T_i, F_out)

            else:
                # (F,) - 1D vector output: direct decoding from latent
                out = self.decoders[i](z)            # (B, F)

            outputs.append(out)

        return outputs