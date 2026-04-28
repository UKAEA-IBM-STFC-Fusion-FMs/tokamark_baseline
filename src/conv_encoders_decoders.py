import torch.nn as nn
import numpy as np

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
    def __init__(self, input_shape, D, layers, kernel_size, stride, padding, bb_factor=3):
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
            D * bb_factor
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
    def __init__(self, output_shape, D, layers=2, kernel_size=3, stride=2, padding=1, bb_factor=3, output_padding=1):

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

        self.fc = nn.Linear(D * bb_factor, final_channels * self.list_ts_comp[0])

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
    def __init__(self, input_shape, D, layers=2, kernel_size=3, stride=2, padding=1, bb_factor=3):

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
            D * bb_factor
        )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        # print('\n in cnn 2D encoder ', x.shape)
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            # print(f'after cnn {i} ', x.shape)
        x = x.flatten(start_dim=1)
        # print('after flatten ', x.shape)
        x = self.fc(x)
        # print('after fc ', x.shape)
        return x


# ======================================================================================================================
class Conv2DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D, layers, kernel_size, stride, padding, bb_factor=3, output_padding=1):

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

        self.fc = nn.Linear(D * bb_factor, final_channels * self.list_ts_comp[0] * self.list_height_comp[0])

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
    def __init__(self, input_shape, D, layers, kernel_size, stride, padding, bb_factor=3):

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
            D * bb_factor
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
    def __init__(self, output_shape, D, layers, kernel_size, stride, padding, bb_factor=3, output_padding=1):

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

        self.fc = nn.Linear(D * bb_factor, final_channels * self.list_ts_comp[0] * self.list_height_comp[0] * self.list_weight_comp[0])

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

