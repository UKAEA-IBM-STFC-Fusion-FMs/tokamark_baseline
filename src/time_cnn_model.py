import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.checkpoint import checkpoint

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
def create_cnn_architecture(dataloader_, D, verbose=True):

    if verbose:
        print("\n\n----------CNN MODEL INITIALIZATION----------\n")

    for l, first_window in enumerate(dataloader_.dataset):
        print(l)
        print(first_window['shot_id'])
        try:

            input_shapes = [arr.shape for arr in first_window["x"]]
            output_shape = [arr.shape for arr in first_window["y"]]

            if verbose:
                print(f"Shot {first_window['shot_id']} used as reference")
                print(f"Input shapes are: {input_shapes}")
                print(f"Output shape are: {output_shape}")

            break  # stop after first successful shot

        except Exception as e:
            print(
                f"Skipping sample {l} because not trainable: {e}"
            )
            continue

    cnn_model = MultiBranchTimeCNNModel(input_shapes, output_shape, D).to(device)

    input_size = [ (2,) + shape for shape in input_shapes ]
    summary(cnn_model, input_size=input_size)

    return cnn_model


# ----------------------------------------------------------------------------------------------------------------------
def compute_compressed_size_encoder(L, layers, kernel_size, stride, padding):
    for _ in range(layers):
        L = (L + 2*padding - kernel_size) // stride + 1
        L = (L + 2*padding - 2) // 2 + 1
    return L


# ----------------------------------------------------------------------------------------------------------------------
def compute_list_compressed_size_decoder(L_out, layers, kernel_size, stride, padding, output_padding=0):

    L = L_out
    list_L = [L]

    for _ in range(layers):
        L = (L + 2*padding - kernel_size + output_padding) // stride + 1 
        L = int(np.ceil(L))
        list_L.append(L)
    return list_L[::-1]


# ======================================================================================================================
class Conv1DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D, layers, kernel_size, stride, padding):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]

        self.ts_comp = compute_compressed_size_encoder(
            self.ts_var, layers, kernel_size, stride, padding
        )
        # print('ts_comp', self.ts_comp)

        modules = []
        in_channels = self.n_var
        out_channels = D

        # initial BN
        modules.append(nn.BatchNorm1d(in_channels))

        for i in range(layers):
            modules.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool1d(2, padding=padding))
            modules.append(nn.BatchNorm1d(out_channels))

            in_channels = out_channels
            out_channels *= 2

        self.cnn = nn.Sequential(*modules)

        # final channels after loop
        final_channels = D * (2 ** (layers - 1))
        # print('final_channels', final_channels)

        self.fc = nn.Linear(
            final_channels * self.ts_comp,
            D
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        for i, layer in enumerate(self.cnn):
            x = layer(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

    
# ======================================================================================================================
class Conv1DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D, layers=2, kernel_size=3, stride=2, padding=1, output_padding=1):

        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]

        self.list_ts_comp = compute_list_compressed_size_decoder(
            self.ts_var, layers, kernel_size, stride, padding, output_padding
        )

        # final encoder channels
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(D, final_channels * self.list_ts_comp[0])

        modules = []
        in_channels = final_channels

        for i in range(layers):
            out_channels = in_channels // 2 if i < layers-1 else self.n_var

            modules.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )

            if i < layers-1:
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm1d(out_channels))

            in_channels = out_channels

        self.transposecnn = nn.Sequential(*modules)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        
        x = self.fc(x)
        
        x = x.view(-1,
                   self.D * (2 ** (self.layers - 1)),
                   self.list_ts_comp[0])

        conv_id = 0  # counts ConvTranspose1d layers only

        for i, layer in enumerate(self.transposecnn):
            x = layer(x)
            # print(f"After layer {layer}: {x.shape}")
            if isinstance(layer, nn.ConvTranspose1d):
                conv_id += 1
                target_ts = self.list_ts_comp[conv_id]  # next expected length
                x = x[:, :, :target_ts]
                # print(f"After ConvTranspose crop {conv_id}: {x.shape}")

        return x


# ======================================================================================================================
class Conv2DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D, layers=2, kernel_size=3, stride=2, padding=1):

        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]
        self.height_var = input_shape[2]

        self.ts_comp = compute_compressed_size_encoder(
            self.ts_var, layers, kernel_size, stride, padding
        )
        self.height_comp = compute_compressed_size_encoder(
            self.height_var, layers, kernel_size, stride, padding
        )

        modules = []
        in_channels = self.n_var
        out_channels = D

        # initial BN
        modules.append(nn.BatchNorm2d(in_channels))

        for i in range(layers):
            modules.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(2, padding=padding))
            modules.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels
            out_channels *= 2

        self.cnn = nn.Sequential(*modules)

        # final channels after loop
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(
            final_channels * self.ts_comp * self.height_comp,
            D
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        for i, layer in enumerate(self.cnn):
            x = layer(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


# ======================================================================================================================
class Conv2DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D, layers, kernel_size, stride, padding, output_padding=1):

        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]
        self.height_var = output_shape[2]

        self.list_ts_comp = compute_list_compressed_size_decoder(
            self.ts_var, layers, kernel_size, stride, padding, output_padding
        )

        self.list_height_comp = compute_list_compressed_size_decoder(
            self.height_var, layers, kernel_size, stride, padding, output_padding
        )

        # final encoder channels
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(D, final_channels * self.list_ts_comp[0] * self.list_height_comp[0])

        modules = []
        in_channels = final_channels

        for i in range(layers):
            out_channels = in_channels // 2 if i < layers-1 else self.n_var

            modules.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )

            if i < layers-1:
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels

        self.transposecnn = nn.Sequential(*modules)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):

        x = self.fc(x)
        x = x.view(-1,
                   self.D * (2 ** (self.layers - 1)),
                   self.list_ts_comp[0], 
                   self.list_height_comp[0])

        conv_id = 0  # counts ConvTranspose2d layers only

        for i, layer in enumerate(self.transposecnn):
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d):
                conv_id += 1
                target_ts = self.list_ts_comp[conv_id]  # next expected length
                target_height = self.list_height_comp[conv_id]
                x = x[:, :, :target_ts, :target_height]

        return x


# ======================================================================================================================
class Conv3DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D, layers, kernel_size, stride, padding):

        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]
        self.height_var = input_shape[2]
        self.weight_var = input_shape[3]

        self.ts_comp = compute_compressed_size_encoder(
            self.ts_var, layers, kernel_size, stride, padding
        )
        self.height_comp = compute_compressed_size_encoder(
            self.height_var, layers, kernel_size, stride, padding
        )
        self.weight_comp = compute_compressed_size_encoder(
            self.weight_var, layers, kernel_size, stride, padding
        )

        modules = []
        in_channels = self.n_var
        out_channels = D

        # initial BN
        modules.append(nn.BatchNorm3d(in_channels))

        for i in range(layers):
            modules.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool3d(2, padding=padding))
            modules.append(nn.BatchNorm3d(out_channels))

            in_channels = out_channels
            out_channels *= 2

        self.cnn = nn.Sequential(*modules)

        # final channels after loop
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(
            final_channels * self.ts_comp * self.height_comp * self.weight_comp,
            D
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        for i, layer in enumerate(self.cnn):
            x = layer(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


# ======================================================================================================================
class Conv3DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D, layers, kernel_size, stride, padding, output_padding=1):

        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]
        self.height_var = output_shape[2]
        self.weight_var = output_shape[3]

        self.list_ts_comp = compute_list_compressed_size_decoder(
            self.ts_var, layers, kernel_size, stride, padding, output_padding
        )

        self.list_height_comp = compute_list_compressed_size_decoder(
            self.height_var, layers, kernel_size, stride, padding, output_padding
        )

        self.list_weight_comp = compute_list_compressed_size_decoder(
            self.weight_var, layers, kernel_size, stride, padding, output_padding
        ) 

        # final encoder channels
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(D, final_channels * self.list_ts_comp[0] * self.list_height_comp[0] * self.list_weight_comp[0])

        modules = []
        in_channels = final_channels

        for i in range(layers):
            out_channels = in_channels // 2 if i < layers-1 else self.n_var

            modules.append(
                nn.ConvTranspose3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding
                )
            )

            if i < layers-1:
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm3d(out_channels))

            in_channels = out_channels

        self.transposecnn = nn.Sequential(*modules)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):

        x = self.fc(x)
        x = x.view(
            -1,
            self.D * (2 ** (self.layers - 1)),
            self.list_ts_comp[0],
            self.list_height_comp[0],
            self.list_weight_comp[0],
            )
        
        conv_id = 0  # counts ConvTranspose3d layers only

        for i, layer in enumerate(self.transposecnn):
            x = layer(x)
            # print(f"After layer {layer}: {x.shape}")
            if isinstance(layer, nn.ConvTranspose3d):
                conv_id += 1
                target_ts = self.list_ts_comp[conv_id]  # next expected length
                target_height = self.list_height_comp[conv_id]
                target_weight = self.list_weight_comp[conv_id]
                x = x[:, :, :target_ts, :target_height, :target_weight]

        return x


# ======================================================================================================================
class MultiBranchTimeCNNModel(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shapes, output_shapes, D=16):
        super().__init__()

        self.D = D
        self.input_branches = nn.ModuleList()

        for shape in input_shapes:
            if len(shape) == 4:  # e.g., (2, T, 15, 17) images evolving in time
                branch = Conv3DEncoder(shape, D, layers_encoder, kernel_size, stride, padding)
            elif len(shape) == 3:  # e.g., (1, T, 15) profiles evolving in time
                branch = Conv2DEncoder(shape, D, layers_encoder, kernel_size, stride, padding)
            elif len(shape) == 2:  # e.g., (7, T, ) time series evolving in time
                branch = Conv1DEncoder(shape, D, layers_encoder, kernel_size, stride, padding)
            else:
                raise ValueError(f"Unsupported input shape: {shape}")
            self.input_branches.append(branch)
        
        self.backbone = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.D*len(self.input_branches), 4*self.D),
            nn.ReLU(),
            nn.Linear(4*self.D, 2*self.D),
            nn.ReLU(),
            nn.Linear(2*self.D, self.D),
            nn.ReLU(),
        )
        
        self.output_branches = nn.ModuleList()

        for var_shape in output_shapes:
            if len(var_shape) == 4:
                branch = Conv3DDecoder(var_shape, D, layers_decoder, kernel_size, stride, padding)
            elif len(var_shape) == 3:
                branch = Conv2DDecoder(var_shape, D, layers_decoder, kernel_size, stride, padding)
            elif len(var_shape) == 2:
                branch = Conv1DDecoder(var_shape, D, layers_decoder, kernel_size, stride, padding)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape}")
            self.output_branches.append(branch)

    # ------------------------------------------------------------------------------------------------------------------
    def _run_encoder(self, branch, x):
        return branch(x)

    # ------------------------------------------------------------------------------------------------------------------
    def _run_decoder(self, branch, x):
        return branch(x)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, *inputs):
        branch_outputs = []

        for branch, x in zip(self.input_branches, inputs):
            out = checkpoint(self._run_encoder, branch, x, use_reentrant=False)
            branch_outputs.append(out)

        merged = torch.cat(branch_outputs, dim=1)

        # checkpoint backbone
        merged = checkpoint(self.backbone, merged, use_reentrant=False)

        decoded_representation = []

        for branch in self.output_branches:
            out = checkpoint(self._run_decoder, branch, merged, use_reentrant=False)
            decoded_representation.append(out)

        return decoded_representation