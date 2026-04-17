
import torch
import torch.nn as nn
import numpy as np 
from torch.utils.checkpoint import checkpoint
from torchinfo import summary

from src.lstm_transform import _resample
from src.time_cnn_model import Conv1DEncoder, Conv2DEncoder, Conv3DEncoder, Conv1DDecoder, Conv2DDecoder, Conv3DDecoder
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

# ----------------------------------------------------------------------------------------------------------------------
def create_lstm_architecture(dataloader_, D, dict_metadata, verbose=True,
                             num_window=32, stride=16):
    
    if verbose:
        print("\n\n----------LSTM MODEL INITIALIZATION----------\n")

    # ------------------------------------------------------------
    # 1. Extract one valid sample for shape inference
    # ------------------------------------------------------------
    for l, first_window in enumerate(dataloader_.dataset):
        print(l)
        print(first_window['shot_id'])

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
    model = MultiBranchTimeCNNModel(
        input_shapes,
        exogenous_shapes,
        output_shapes,
        dict_metadata=dict_metadata,
        D=D,
    ).to(device)

    # ------------------------------------------------------------
    # 3. Build dummy input sizes for torchinfo
    #    IMPORTANT: we only pass raw tensors BEFORE windowing
    # ------------------------------------------------------------
    input_sizes = []

    for shape in input_shapes:
        # shape is (C, T) or (C, T, H) etc.
        # we add batch dimension only
        input_sizes.append((2,) + shape)

    # ------------------------------------------------------------
    # 4. Model summary
    # ------------------------------------------------------------
    summary(model, input_size=input_sizes)

    return model

# ======================================================================================================================
import numpy as np

def make_dummy_outputs(output_shapes, dict_metadata):

    lstm_dt = max(
        dict_metadata[section][var]['dt']
        for section in ['input', 'actuator', 'output']
        for var in dict_metadata[section]
    )

    shot_section = {}

    for var, shape in zip(dict_metadata['output'].keys(), output_shapes):

        print(var, shape)
        # shape = (T, ...)
        T = shape[0]

        # create time axis
        time = np.arange(T)

        # create values
        values = np.random.randn(*shape)

        shot_section[var] = {
            "time": time,
            "values": values
        }
    
    y = _resample(shot_section, dict_metadata['output'], lstm_dt)
    y = [np.expand_dims(arr, axis=1) for arr in y]

    return y

# ======================================================================================================================
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden, batch_first=True)

#     def forward(self, x):
#         _, (h, c) = self.lstm(x)
#         return h, c


# class Decoder(nn.Module):
#     def __init__(self, input_size, hidden, output_size):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
#         self.fc = nn.Linear(hidden, output_size)

#     def forward(self, x, hidden):
#         out, _ = self.lstm(x, hidden)
#         return self.fc(out)


# class Seq2Seq(nn.Module):
#     def __init__(self, enc_in, dec_in, hidden, out_dim):
#         super().__init__()
#         self.encoder = Encoder(enc_in, hidden)
#         self.decoder = Decoder(dec_in, hidden, out_dim)

#     def forward(self, enc_x, dec_x):
#         h, c = self.encoder(enc_x)
#         return self.decoder(dec_x, (h, c))

# ======================================================================================================================
class MultiBranchTimeCNNModel(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, 
                    input_shapes, 
                    exogenous_shapes,
                    output_shapes,
                    dict_metadata,
                    D=16
                ):

        super().__init__()

        self.D = D

        # --------------------------------------------------------------------------------------------------------------
        self.input_branches = nn.ModuleList()

        for var_shape in input_shapes:
            if len(var_shape) == 5:  # e.g., (2, T, 15, 17) images evolving in time
                branch = Conv3DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding)
            elif len(var_shape) == 4:  # e.g., (1, T, 15) profiles evolving in time
                branch = Conv2DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding)
            elif len(var_shape) == 3:  # e.g., (7, T, ) time series evolving in time
                branch = Conv1DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.input_branches.append(branch)

        # --------------------------------------------------------------------------------------------------------------
        self.backbone = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.D*len(self.input_branches), 4*self.D),
            nn.ReLU(),
            nn.Linear(4*self.D, 2*self.D),
            nn.ReLU(),
            nn.Linear(2*self.D, self.D),
            nn.ReLU(),
        )

        self.encoder_lstm = nn.LSTM(
            input_size=self.D * len(self.input_branches),
            hidden_size=self.D,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.decoder_lstm = nn.LSTM(
            input_size=self.D,
            hidden_size=self.D,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # --------------------------------------------------------------------------------------------------------------
        self.exogenous_branches = nn.ModuleList()

        for var_shape in exogenous_shapes:
            if len(var_shape) == 5:  # e.g., (2, T, 15, 17) images evolving in time
                branch = Conv3DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding)
            elif len(var_shape) == 4:  # e.g., (1, T, 15) profiles evolving in time
                branch = Conv2DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding)
            elif len(var_shape) == 3:  # e.g., (7, T, ) time series evolving in time
                branch = Conv1DEncoder(var_shape[1:], D, layers_encoder, kernel_size, stride, padding)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.exogenous_branches.append(branch)

        # --------------------------------------------------------------------------------------------------------------
        self.output_branches = nn.ModuleList()

        self.output_shapes = output_shapes
        y = make_dummy_outputs(output_shapes, dict_metadata)
        output_latent_shapes = [arr.shape for arr in y]
        print(output_latent_shapes)
        self.W_out = output_latent_shapes[0][0]

        for var_shape in output_latent_shapes:

            if len(var_shape) == 5:
                branch = Conv3DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding)
            elif len(var_shape) == 4:
                branch = Conv2DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding)
            elif len(var_shape) == 3:
                branch = Conv1DDecoder(var_shape[1:], D, layers_decoder, kernel_size, stride, padding)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape[1:]}")
            self.output_branches.append(branch)

    # # ------------------------------------------------------------------------------------------------------------------
    # def _spaced_windows_tensor(self, arr, n, L):
    #     N = arr.shape[0]
    #     if L > N:
    #         raise ValueError("L is larger than first dimension")

    #     max_start = N - L

    #     if n == 1:
    #         starts = np.array([0])
    #     else:
    #         starts = np.linspace(0, max_start, n)
    #         starts = np.round(starts).astype(int)

    #     windows = np.stack([arr[s:s+L] for s in starts], axis=0)
    #     return windows

    # # ------------------------------------------------------------------------------------------------------------------
    # def _resample(self, shot_section, metadata_section, t_cut):
    #     resampled = []

    #     for var, shot in shot_section.items():
    #         time = shot["time"]
    #         values = shot["values"]

    #         dt_var = metadata_section[var]['dt']
    #         w_size = int(round(self.lstm_dt / dt_var))
    #         n_window = int( len(time) / w_size )

    #         if w_size <= 0:
    #             raise ValueError(f"Invalid resampling factor for {var}")

    #         # Reshape windows
    #         # print(f' {var} before :', values.shape)
    #         resampled_values = self._spaced_windows_tensor(values, n_window, w_size)
    #         # print(f' {var} after :', resampled_values.shape)

    #         resampled.append(resampled_values)

    #     return resampled

    # ------------------------------------------------------------------------------------------------------------------
    def _run_cnn_encoder(self, branch, x):
        # print(' reshape before cnn', x.shape)
        B, W = x.shape[:2]  # batch, num_windows
        # merge batch and window dims
        x = x.view(B * W, *x.shape[2:])
        # pass through CNN
        # print('before cnn', x.shape)
        out = branch(x)
        # print('after cnn', out.shape)
        # restore dimensions
        out = out.view(B, W, -1)
        # optionally aggregate windows (VERY important design choice)
        # out = out.mean(dim=1)  # or sum / max / keep all
        # print('after cnn', out.shape)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    def _run_cnn_decoder(self, branch, goal_shape, x):

        # print('shape reshape decoder', x.shape)
        B, W = x.shape[:2]  # batch, num_windows
        x = x.reshape(B * W, *x.shape[2:])
        # print('before cnn decoder', x.shape)
        out = branch(x)
        # print('after cnn decoder', out.shape)
        # print('but goal shape is ', goal_shape)
        out = out.reshape(B, W * out.shape[2], *out.shape[3:])
        # print('after batch reshape ', out.shape)

        return out

    # ------------------------------------------------------------------------------------------------------------------
    # def forward(self, *inputs):
    #     branch_outputs = []

    #     for branch, x in zip(self.input_branches, inputs):
    #         out = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
    #         branch_outputs.append(out)

    #     merged = torch.cat(branch_outputs, dim=2)

    #     enc_out, (h, c) = self.encoder_lstm(merged) # enc_out: (B, W_in, D)


    #     # checkpoint backbone
    #     merged = checkpoint(self.backbone, merged, use_reentrant=False)

    #     decoded_representation = []

    #     for branch in self.output_branches:
    #         out = checkpoint(self._run_cnn_decoder, branch, merged, use_reentrant=False)
    #         decoded_representation.append(out)

    #     return decoded_representation

    # def forward(self, inputs, exogenous):
    #     branch_outputs = []

    #     # ------------------------------------------------------------------
    #     # 1. Encode each input branch (CNN encoders)
    #     # ------------------------------------------------------------------
    #     for branch, x in zip(self.input_branches, inputs):
    #         encoded_input = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
    #         branch_outputs.append(encoded_input)

    #     # ------------------------------------------------------------------
    #     # 1bis. Encode each exogenous future branch (CNN encoders)
    #     # ------------------------------------------------------------------
    #     for branch, x in zip(self.exogenous_branches, inputs):
    #         encoded_exogenous = checkpoint(self._run_cnn_encoder, branch, x, use_reentrant=False)
    #         branch_outputs.append(encoded_exogenous)

    #     # (B, W_in, D * num_branches)
    #     merged_input = torch.cat(branch_outputs, dim=2)

    #     # ------------------------------------------------------------------
    #     # 2. Temporal encoding (LSTM encoder)
    #     # ------------------------------------------------------------------
    #     enc_out, (h, c) = self.encoder_lstm(merged_input)
    #     # enc_out: (B, W_in, D)

    #     # ------------------------------------------------------------------
    #     # 3. Build decoder input (Seq2Seq)
    #     # ------------------------------------------------------------------

    #     B = enc_out.size(0)

    #     # Option A: repeat last hidden state across W_out
    #     context = enc_out[:, -1:, :]  # (B, 1, D)
    #     decoder_input = context.repeat(1, self.W_out, 1)

    #     # If you don't have self.W_out:
    #     # decoder_input = self.start_token.repeat(B, W_out, 1)

    #     # ------------------------------------------------------------------
    #     # 4. Temporal decoding (LSTM decoder)
    #     # ------------------------------------------------------------------
    #     dec_out, _ = self.decoder_lstm(decoder_input, (h, c))
    #     # dec_out: (B, W_out, D)

    #     # ------------------------------------------------------------------
    #     # 5. Decode each timestep through output branches
    #     # ------------------------------------------------------------------
    #     decoded_representation = []

    #     for branch in self.output_branches:
    #         out = checkpoint(self._run_cnn_decoder, branch, dec_out, use_reentrant=False)
    #         decoded_representation.append(out)

    #     return decoded_representation


    def forward(self, *args):

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
        # encoder
        # -----------------------------
        enc_out, (h, c) = self.encoder_lstm(input_seq)

        # -----------------------------
        # decoder conditioning
        # -----------------------------
        context = enc_out[:, -1:, :]
        context_seq = context.repeat(1, self.W_out, 1)

        if exo_seq is not None:
            decoder_input = torch.cat([context_seq, exo_seq], dim=2)
        else:
            decoder_input = context_seq

        # -----------------------------
        # decoder
        # -----------------------------
        dec_out, _ = self.decoder_lstm(decoder_input, (h, c))

        # -----------------------------
        # output heads
        # -----------------------------
        outputs = []
        for branch, goal_shape in zip(self.output_branches, self.output_shapes):
            out = checkpoint(self._run_cnn_decoder, branch, goal_shape, dec_out, use_reentrant=False)
            outputs.append(out)

        return outputs