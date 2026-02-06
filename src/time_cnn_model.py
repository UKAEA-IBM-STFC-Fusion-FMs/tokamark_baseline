import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Set device
from MAST_benchmark.tools.utils import get_device
device = get_device()
# print(f"Using device: {device}\n")


padding = 1
kernel_size = 3
stride = 3
layers_encoder = 3
layers_decoder = 3


# ----------------------------------------------------------------------------------------------------------------------
def create_cnn_architecture(dataloader_, D, verbose=False):
    if verbose:
        print("\n\n----------CNN MODEL INITIALIZATION----------\n")

    for l in range(len(dataloader_.dataset)):
        try:
            windows_gen = dataloader_.dataset[l]  # this is a generator
            first_window = next(windows_gen) 
            input_shapes = [arr.shape for arr in first_window["x"]]
            output_shape = [arr.shape for arr in first_window["y"]]

            if verbose:
                print(f"Shot {dataloader_.dataset.get_shot_id(l)} used as reference")
                print(f"Input shapes are: {input_shapes}")
                print(f"Output shape are: {output_shape}")

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

def compute_compressed_size_encoder(L, layers, kernel_size, stride, padding):
    for _ in range(layers):
        L = (L + 2*padding - kernel_size) // stride + 1
        L = (L + 2*padding - 2) // 2 + 1
    return L

def compute_list_compressed_size_decoder(L_out, layers, kernel_size, stride, padding, output_padding=0):

    L = L_out
    list_L = [L]

    for _ in range(layers):
        L = (L + 2*padding - kernel_size + output_padding) // stride + 1 
        L = int( np.ceil( L ) )
        list_L.append(L)
    return list_L[::-1]


# ======================================================================================================================
class Conv1DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D, layers,
                 kernel_size, stride, padding):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]

        self.ts_comp = compute_compressed_size_encoder(
            self.ts_var, layers, kernel_size, stride, padding
        )
        # # print('ts_comp', self.ts_comp)

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
        # # print('final_channels', final_channels)

        self.fc = nn.Linear(
            final_channels * self.ts_comp,
            D
        )

    def forward(self, x):
        # x = self.cnn(x)
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            # # print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # # print('After flatten:', x.shape)
        x = self.fc(x)
        # # print('After FC:', x.shape)
        return x

    
# ======================================================================================================================
class Conv1DDecoder(nn.Module):

    def __init__(self, output_shape, D, layers=2,
                 kernel_size=3, stride=2, padding=1, output_padding=1):
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
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(D, final_channels * self.list_ts_comp[0]),
        #     nn.ReLU()
        # )

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

    def forward(self, x):
        
        # print('In Decoder: ', x.shape)
        x = self.fc(x)
        # print('After FC: ', x.shape)
        
        x = x.view(-1,
                   self.D * (2 ** (self.layers - 1)),
                   self.list_ts_comp[0])
        # print('After reshape: ', x.shape)

        conv_id = 0  # counts ConvTranspose1d layers only

        for i, layer in enumerate(self.transposecnn):
            x = layer(x)
            # print(f"After layer {layer}: {x.shape}")
            if isinstance(layer, nn.ConvTranspose1d):
                conv_id += 1
                target_ts = self.list_ts_comp[conv_id]  # next expected length
                x = x[:, :, :target_ts]
                # print(f"After ConvTranspose crop {conv_id}: {x.shape}")

        # crop/pad to original length
        # print('Out shape : ', x.shape)

        return x



# ======================================================================================================================
class Conv2DEncoder(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D, layers=2,
                 kernel_size=3, stride=2, padding=1):
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

        # # print('ts_comp', self.ts_comp)
        # # print('height_comp', self.height_comp)

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
        # # print('final_channels', final_channels)

        self.fc = nn.Linear(
            final_channels * self.ts_comp * self.height_comp,
            D
        )

    def forward(self, x):
        # x = self.cnn(x)
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            # # print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # # print('After flatten:', x.shape)
        x = self.fc(x)
        # # print('After FC:', x.shape)
        return x


# ======================================================================================================================
class Conv2DDecoder(nn.Module):
    def __init__(self, output_shape, D, layers,
                 kernel_size, stride, padding, output_padding=1):
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

        # print('ts_comp', self.ts_comp)
        # print('height_comp', self.height_comp)

        # final encoder channels
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(D, final_channels * self.list_ts_comp[0] * self.list_height_comp[0])
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(D, final_channels * self.list_ts_comp[0] * self.list_height_comp[0]),
        #     nn.ReLU()
        # )

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

    def forward(self, x):

        # print('In Decoder: ', x.shape)
        x = self.fc(x)
        # print('After FC: ', x.shape)
        x = x.view(-1,
                   self.D * (2 ** (self.layers - 1)),
                   self.list_ts_comp[0], 
                   self.list_height_comp[0])
        # print('After reshape: ', x.shape)

        conv_id = 0  # counts ConvTranspose2d layers only

        for i, layer in enumerate(self.transposecnn):
            x = layer(x)
            # print(f"After layer {layer}: {x.shape}")
            if isinstance(layer, nn.ConvTranspose2d):
                conv_id += 1
                target_ts = self.list_ts_comp[conv_id]  # next expected length
                target_height = self.list_height_comp[conv_id]
                x = x[:, :, :target_ts, :target_height]
                # print(f"After ConvTranspose crop {conv_id}: {x.shape}")

        # print('Out shape : ', x.shape)

        return x

# ======================================================================================================================
class Conv3DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D, layers,
                 kernel_size, stride, padding):
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

        # # print('ts_comp', self.ts_comp)
        # # print('height_comp', self.height_comp)
        # # print('weight_comp', self.weight_comp)

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
        # # print('final_channels', final_channels)

        self.fc = nn.Linear(
            final_channels * self.ts_comp * self.height_comp * self.weight_comp,
            D
        )

    def forward(self, x):
        # x = self.cnn(x)
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            # # print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # # print('After flatten:', x.shape)
        x = self.fc(x)
        # # print('After FC:', x.shape)
        return x


# ======================================================================================================================
class Conv3DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D, layers,
                 kernel_size, stride, padding, output_padding=1):
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

        # print('ts_comp', self.ts_comp)
        # print('height_comp', self.height_comp)
        # print('weight_comp', self.weight_comp)

        # final encoder channels
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(D, final_channels * self.list_ts_comp[0] * self.list_height_comp[0] * self.list_weight_comp[0])
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(D, final_channels * self.list_ts_comp[0] * self.list_height_comp[0] * self.list_weight_comp[0]),
        #     nn.ReLU()
        # )

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

    def forward(self, x):

        # print('In Decoder: ', x.shape)
        x = self.fc(x)
        # print('After FC: ', x.shape)
        x = x.view(
            -1,
            self.D * (2 ** (self.layers - 1)),
            self.list_ts_comp[0],
            self.list_height_comp[0],
            self.list_weight_comp[0],
            )
        # print('After reshape: ', x.shape)
        
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
                # print(f"After ConvTranspose crop {conv_id}: {x.shape}")

        # print('Out shape : ', x.shape)

        return x

# ======================================================================================================================
class MultiBranchTimeCNNModel(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shapes, output_shapes, D=16):
        super().__init__()

        self.D = D
        self.input_branches = nn.ModuleList()
        # merged_dim = 0

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
            # print('shape', var_shape, 'len', len(var_shape))
            if len(var_shape) == 4:
                branch = Conv3DDecoder(var_shape, D, layers_decoder, kernel_size, stride, padding)
            elif len(var_shape) == 3:
                branch = Conv2DDecoder(var_shape, D, layers_decoder, kernel_size, stride, padding)
            # elif len(var_shape) == 2 and var_shape != (1, 2):
            elif len(var_shape) == 2:
                branch = Conv1DDecoder(var_shape, D, layers_decoder, kernel_size, stride, padding)
            # elif var_shape == (1, 2):
            #     # Flatten first if needed, like 1D input
            #     # print('x_point exception')
            #     branch = nn.Sequential(
            #         nn.Flatten(),              # (1, 2) -> (2,)
            #         nn.Linear(D, 2*D),         # same structure as 0D branch
            #         nn.ReLU(),
            #         nn.Linear(2*D, 4*D),
            #         nn.ReLU(),
            #         nn.Linear(4*D, 2),         
            #         nn.Unflatten(1, var_shape) # reshape back to (1, 2)
            #     )
            # elif len(var_shape) == 1:
            #     print(var_shape)
            #     branch = nn.Sequential(
            #         nn.Linear(D, 2*D),
            #         nn.ReLU(),
            #         nn.Linear(2*D, 4*D),
            #         nn.ReLU(),
            #         nn.Linear(4*D, var_shape[0])
            #     )
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
            out = checkpoint(self._run_encoder, branch, x)
            branch_outputs.append(out)

        merged = torch.cat(branch_outputs, dim=1)

        # checkpoint backbone
        merged = checkpoint(self.backbone, merged)

        decoded_representation = []

        for branch in self.output_branches:
            out = checkpoint(self._run_decoder, branch, merged)
            decoded_representation.append(out)

        return decoded_representation


# ======================================================================================================================
from torchinfo import summary

D = 16
B = 6

tasks = {
    "1-1": {
        "input": [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18),
                (1, 250, 12), (1, 20, 10), (1, 20), (1, 20)],
        "output": [(1,), (1,), (1,), (1,), (1, 2), (1, 2),
                (1,), (1,), (1,), (1,), (1,), (1,), 
                (1,), (1,), (1,)]
    },
    "1-2": {
        "input": [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18),
                  (1, 250, 12), (1, 20, 10), (1, 20), (1, 20)],
        "output": [(1, 1, 170), (1, 1, 170)]
    },
    "1-3": {
        "input": [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18),
                  (1, 250, 12), (1, 20, 10), (1, 20), (1, 20)],
        "output": [(1, 1, 65, 65)]
    },
    "2-1": {
        "input": [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18),
                  (1, 250, 12), (1, 20, 10), (1, 20), (1, 20),
                  (1, 120, 4), (1, 120)],
        "output": [(1, 100, 10), (1, 100), (1, 100), (1, 5), (1, 5), 
                   (1, 5), (1, 5), (1, 5, 2), (1, 5, 2), (1, 5), 
                   (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), (1, 5), 
                   (1, 5), (1, 5)]
    },
    "2-2": {
        "input": [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18),
                  (1, 250, 12), (1, 20, 10), (1, 20), (1, 20),
                  (1, 120, 4), (1, 120)],
        "output": [(1, 5, 170), (1, 5, 170)]
    },
    "2-3": {
        "input": [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18),
                  (1, 250, 12), (1, 20, 10), (1, 20), (1, 20),
                  (1, 120, 4), (1, 120)],
        "output": [(1, 5, 65, 65)]
    },
    "3-1": {
        "input": [(1, 1, 120), (1, 1, 120), (1, 220), (1, 220),
                  (1, 220), (1, 220)],
        "output": [(1, 10, 120), (1, 10, 120)]
    },
    "3-2": {
        "input": [(1, 1, 120), (1, 1, 120), (1, 250, 3), (1, 250, 18),
                  (1, 250, 18), (1, 220), (1, 220), (1, 220), (1, 220)],
        "output": [(1, 2500, 3), (1, 2500, 18), (1, 2500, 18)]
    },
    "3-3": {
        "input": [(1, 750, 15), (1, 750, 40), (1, 750, 18), (1, 750, 18),
                  (1, 7500, 12), (1, 600, 10), (1, 600), (1, 600), (1, 600),
                  (1, 620), (1, 620), (1, 620), (1, 620)],
        "output": [(1, 1, 120), (1, 1, 120), (1, 1), (1, 1), (1, 1)]
    },
    "4-1": {
        "input": [(1, 750, 15), (1, 750, 40), (1, 750, 18), (1, 750, 18),
                  (1, 7500, 12), (1, 600, 10), (1, 600), (1, 600), (1, 600),
                  (1, 7500, 3), (1, 7500, 18), (1, 7500, 18),
                  (1, 75000, 3), (1, 75000, 3), (1, 1000), (1, 1000),
                  (1, 1000), (1, 1000)],
        "output": [(1, 5000, 18), (1, 5000, 18)]

    },
    "4-2": {
        "input": [(1, 750, 15), (1, 750, 40), (1, 750, 18), (1, 750, 18),
                  (1, 7500, 12), (1, 600, 10), (1, 600), (1, 600), (1, 600),
                  (1, 7500, 3), (1, 7500, 18), (1, 7500, 18),
                  (1, 75000, 3), (1, 75000, 3), (1, 30, 120), (1, 30, 120),
                  (1, 1000), (1, 1000), (1, 1000), (1, 1000)],
        "output": [(1, 5000, 18), (1, 5000, 18)]
    },
    "4-3": {
        "input": [(1, 750, 15), (1, 750, 40), (1, 750, 18), (1, 750, 18),
                  (1, 7500, 12), (1, 600, 10), (1, 600), (1, 600), (1, 600),
                  (1, 7500, 3), (1, 7500, 18), (1, 7500, 18),
                  (1, 75000, 3), (1, 75000, 3), (1, 1000), (1, 1000),
                  (1, 1000), (1, 1000)],
        "output": [(1, 20)]
    },
    "4-4": {
        "input": [(1, 750, 15), (1, 750, 40), (1, 750, 18), (1, 750, 18),
                  (1, 7500, 12), (1, 600, 10), (1, 600), (1, 600), (1, 600),
                  (1, 7500, 3), (1, 7500, 18), (1, 7500, 18),
                  (1, 75000, 3), (1, 75000, 3), (1, 1000), (1, 1000),
                  (1, 1000), (1, 1000)],
        "output": [(1, 400)]
    },
    "4-5": {
        "input": [(1, 750, 15), (1, 750, 40), (1, 750, 18), (1, 750, 18),
                  (1, 7500, 12), (1, 600, 10), (1, 600), (1, 600), (1, 600),
                  (1, 7500, 3), (1, 7500, 18), (1, 7500, 18),
                  (1, 75000, 3), (1, 75000, 3), (1, 1000), (1, 1000),
                  (1, 1000), (1, 1000)],
        "output": [(1, 50000, 3), (1, 50000, 3)]

    }
}


# results = {}

# for name, cfg in tasks.items():

#     model = MultiBranchTimeCNNModel(cfg["input"], cfg["output"], D)

#     # dummy input
#     dummy_input = [torch.randn((B,)+s) for s in cfg["input"]]

#     # get actual output shapes
#     with torch.no_grad():
#         actual_outputs = model(*dummy_input)
#         actual_shapes = [tuple(o.shape) for o in actual_outputs]

#     # torchinfo summary
#     shape_input = [(B,)+s for s in cfg["input"]]
#     s = summary(model, input_size=(shape_input), verbose=0)
#     print(s)
    
#     results[name] = {
#         "total_params": s.total_params,
#         "trainable_params": s.trainable_params,
#         "non_trainable_params": s.total_params - s.trainable_params,
#         "actual_output_shapes": actual_shapes,
#         "demanded_output_shapes": cfg["output"]
#     }

# # # print RESULTS
# for k,v in results.items():
#     print(f"\nTASK {k}")
#     print(f"Total params: {v['total_params']:,}")
#     print(f"Trainable params: {v['trainable_params']:,}")
#     print(f"Non-trainable params: {v['non_trainable_params']:,}")
#     print("Actual output shapes:")
#     for shp in v['actual_output_shapes']:
#         print("   ", shp)
#     print("Demanded output shapes:")
#     for shp in v['demanded_output_shapes']:
#         print("   ", shp)




