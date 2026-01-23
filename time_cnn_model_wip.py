import torch
# torch.backends.mkldnn.enabled = False
import torch.nn as nn
import numpy as np


padding = 1
kernel_size = 3
stride = 2
layers = 3 #not changing the model yet!!!!

def compute_compressed_size(input_size, layers, kernel_size, stride, padding, output_padding=0):
    size = input_size
    for _ in range(layers):
        size = int(np.ceil((size + 2*padding - kernel_size - output_padding) / stride + 1))
    return size

def compute_compressed_size_encoder(L, layers, kernel_size, stride, padding):

    for _ in range(layers):

        # Conv1D
        L = (L + 2*padding - kernel_size) // stride + 1

        # MaxPool1D (kernel=2, stride=2)
        L = (L + 2*padding - 2) // 2 + 1

    return L

def compute_compressed_size_decoder(L, layers, kernel_size, stride, padding):

    for _ in range(layers):

        # Conv1D
        L = (L + 2*padding - kernel_size) // stride + 1

    return L




# ======================================================================================================================
class Conv1DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D, layers=2,
                 kernel_size=3, stride=2, padding=1):
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

    def forward(self, x):
        # x = self.cnn(x)
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            # print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        # # print(x.shape)
        x = x.flatten(start_dim=1)
        # print('After flatten:', x.shape)
        x = self.fc(x)
        # print('After FC:', x.shape)
        return x

    
# ======================================================================================================================
class Conv1DDecoder(nn.Module):

    def __init__(self, output_shape, D, layers=2,
                 kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]

        self.ts_comp = compute_compressed_size_decoder(
            self.ts_var, layers, kernel_size, stride, padding
        )

        # final encoder channels
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(D, final_channels * self.ts_comp)

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
                    padding=padding
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
                   self.ts_comp)
        # print('After reshape: ', x.shape)

        for i, layer in enumerate(self.transposecnn):
            x = layer(x)
            # print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")

        # crop/pad to original length
        x = x[:, :, :self.ts_var]
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

        # print('ts_comp', self.ts_comp)
        # print('height_comp', self.height_comp)

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
        # print('final_channels', final_channels)

        self.fc = nn.Linear(
            final_channels * self.ts_comp * self.height_comp,
            D
        )

    def forward(self, x):
        # x = self.cnn(x)
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            # print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        # # print(x.shape)
        x = x.flatten(start_dim=1)
        # print('After flatten:', x.shape)
        x = self.fc(x)
        # print('After FC:', x.shape)
        return x


# ======================================================================================================================
class Conv2DDecoder(nn.Module):
    def __init__(self, output_shape, D, layers=2,
                 kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]
        self.height_var = output_shape[2]

        self.ts_comp = compute_compressed_size_decoder(
            self.ts_var, layers, kernel_size[0], stride[0], padding
        )

        self.height_comp = compute_compressed_size_decoder(
            self.height_var, layers, kernel_size[1], stride[1], padding
        )

        print('ts_comp', self.ts_comp)
        print('height_comp', self.height_comp)

        # final encoder channels
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(D, final_channels * self.ts_comp * self.height_comp)

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
                    padding=padding
                )
            )

            if i < layers-1:
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm2d(out_channels))

            in_channels = out_channels

        self.transposecnn = nn.Sequential(*modules)

    def forward(self, x):

        print('In Decoder: ', x.shape)
        x = self.fc(x)
        print('After FC: ', x.shape)
        x = x.view(-1,
                   self.D * (2 ** (self.layers - 1)),
                   self.ts_comp, 
                   self.height_comp)
        print('After reshape: ', x.shape)

        for i, layer in enumerate(self.transposecnn):
            x = layer(x)
            print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")

        # crop/pad to original length
        x = x[:, :, :self.ts_var, :self.height_var]
        print('Out shape : ', x.shape)

        return x

# ======================================================================================================================
class Conv3DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D, layers=2,
                 kernel_size=3, stride=2, padding=1):
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

        # print('ts_comp', self.ts_comp)
        # print('height_comp', self.height_comp)
        # print('weight_comp', self.weight_comp)

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
        # print('final_channels', final_channels)

        self.fc = nn.Linear(
            final_channels * self.ts_comp * self.height_comp * self.weight_comp,
            D
        )

    def forward(self, x):
        # x = self.cnn(x)
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            # print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        # # print(x.shape)
        x = x.flatten(start_dim=1)
        # print('After flatten:', x.shape)
        x = self.fc(x)
        # print('After FC:', x.shape)
        return x


# ======================================================================================================================
class Conv3DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D, layers=2,
                 kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.D = D
        self.layers = layers
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]
        self.height_var = output_shape[2]
        self.weight_var = output_shape[3]

        self.ts_comp = compute_compressed_size_decoder(
            self.ts_var, layers, kernel_size, stride, padding
        )

        self.height_comp = compute_compressed_size_decoder(
            self.height_var, layers, kernel_size, stride, padding
        )

        self.weight_comp = compute_compressed_size_decoder(
            self.weight_var, layers, kernel_size, stride, padding
        )

        # print('ts_comp', self.ts_comp)
        # print('height_comp', self.height_comp)
        # print('weight_comp', self.weight_comp)

        # final encoder channels
        final_channels = D * (2 ** (layers - 1))

        self.fc = nn.Linear(D, final_channels * self.ts_comp * self.height_comp * self.weight_comp)

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
                    padding=padding
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
        x = x.view(-1,
                   self.D * (2 ** (self.layers - 1)),
                   self.ts_comp, 
                   self.height_comp, 
                   self.weight_comp)
        # print('After reshape: ', x.shape)

        for i, layer in enumerate(self.transposecnn):
            x = layer(x)
            # print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")

        # crop/pad to original length
        x = x[:, :, :self.ts_var, :self.height_var, :self.weight_var]
        # print('Out shape : ', x.shape)

        return x

# ======================================================================================================================
class MultiBranchTimeCNNModel(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, 
                 input_shapes, 
                 output_shapes, 
                 D=16):
        super().__init__()

        self.D = D
        self.input_branches = nn.ModuleList()
        # merged_dim = 0

        for shape in input_shapes:
            if len(shape) == 4:  # e.g., (2, T, 15, 17) images evolving in time
                branch = Conv3DEncoder(shape, D, layers, kernel_size, stride, padding)
            elif len(shape) == 3:  # e.g., (1, T, 15) profiles evolving in time
                branch = Conv2DEncoder(shape, D, layers, kernel_size, stride, padding)
            elif len(shape) == 2:  # e.g., (7, T, ) time series evolving in time
                branch = Conv1DEncoder(shape, D, layers, kernel_size, stride, padding)
            else:
                raise ValueError(f"Unsupported input shape: {shape}")
            self.input_branches.append(branch)
        
        self.backbone = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.D*len(self.input_branches), self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.D),
            nn.ReLU(),
        )
        
        self.output_branches = nn.ModuleList()

        for var_shape in output_shapes:
            # # print('shape', var_shape, 'len', len(var_shape))
            if len(var_shape) == 4:
                branch = Conv3DDecoder(var_shape, D, layers, kernel_size, stride, padding)
            elif len(var_shape) == 3:
                branch = Conv2DDecoder(var_shape, D, layers, [kernel_size, kernel_size] , [stride, stride], padding)
            elif len(var_shape) == 2:
                branch = Conv1DDecoder(var_shape, D, layers, kernel_size, stride, padding)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape}")
            self.output_branches.append(branch)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, *inputs):
        branch_outputs = []

        for branch, x in zip(self.input_branches, inputs):
            # print(f"\n Encoder of {x.shape}")
            out = branch(x)
            branch_outputs.append(out)
        
        # # # print('\nCommon Layer to flatten time')     
        merged = torch.cat(branch_outputs, dim=1)
        print(merged.shape)        
        merged = self.backbone(merged)
        print('after backbone', merged.shape)   

        decoded_representation = []

        for branch in self.output_branches :
            # # print("\n ")
            out = branch(merged)
            decoded_representation.append(out)
        
        return decoded_representation 

        # return merged

    # ------------------------------------------------------------------------------------------------------------------



# ======================================================================================================================
# Example usage
# batch_size = 8

# D = 16
# T = 100

# input_channels_1 = 1  # Number of input channels

# input_channels_2 = 1  
# height_length_2 = 54

# input_channels_3 = 1  # Number of input channels
# height_length_3 = 27
# width_length_3 = 33

# output_shape = [[7]]

# x_init = [ torch.randn(input_channels_1, T), # time series evolving in time
#       torch.randn(input_channels_2, T*2, height_length_2), # profiles evolving in time
#       torch.randn(input_channels_3, T+10, height_length_3, width_length_3) # images evolving in time
#       ]

# model = MultiBranchTimeCNNModel([arr.shape for arr in x_init], output_shape, D)

# x = [ torch.randn(batch_size, input_channels_1, T), # time series evolving in time
#       torch.randn(batch_size, input_channels_2, T*2, height_length_2), # profiles evolving in time
#       torch.randn(batch_size, input_channels_3, T+10, height_length_3, width_length_3) # images evolving in time
# ]

# output = model(x)

# ======================================================================================================================
# Example usage

D = 16
B = 2

# input_shapes = [(1, 7), (1, 1), (1, 65), (3, 12, 65), (1, 5, 6, 7), (1, 1, 1, 1)]
# output_shapes = [(1, 1), (1, 200, 5), (1, 50, 15, 6), (1, 200, 15, 65), (1, 1, 1, 1)]

# input_shapes = [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18), (1, 250, 12), (1, 20, 10), (1, 20), (1, 20)]
# output_shapes = [(1, 1), (1, 1), (1, 1)]

# input_shapes = [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18), (1, 250, 12), (1, 20, 10), (1, 20), (1, 20)]
# output_shapes = [(1, 1, 65, 65)]

# input_shapes = [(1, 1, 120), (1, 1, 120), (1, 220), (1, 220), (1, 220), (1, 220)]
# output_shapes = [(1, 10, 120), (1, 10, 120)]

# input_shapes = [(1, 750, 15), (1, 750, 40), (1, 750, 18), (1, 750, 18), (1, 7500, 12), (1, 600, 10), 
#                 (1, 600), (1, 600), (1, 600), (1, 7500, 3), (1, 75000, 18), (1, 75000, 18), (1, 75000, 3), 
#                 (1, 75000, 3), (1, 30, 120), (1, 30, 120), (1, 1000), (1, 1000), (1, 1000), (1, 1000)]
# output_shapes = [(1, 50000, 18), (1, 50000, 18)]

# input_shapes = [(1, 750, 15), (1, 750, 40), (1, 750, 18), (1, 750, 18), (1, 7500, 12), (1, 600, 10), (1, 600), (1, 600), (1, 600), (1, 7500, 3), (1, 75000, 18), (1, 75000, 18), (1, 75000, 3), (1, 75000, 3), (1, 1000), (1, 1000), (1, 1000), (1, 1000)]
# output_shapes = [(1, 20)]

input_shapes = [(1, 1, 120), (1, 1, 120), (1, 250, 3), (1, 250, 18), (1, 250, 18), (1, 220), (1, 220), (1, 220), (1, 220)]
output_shapes = [(1, 2500, 3), (1, 2500, 18), (1, 2500, 18)]
# output_shapes = [(1, 2500, 3), (1, 25000, 18), (1, 25000, 18)]\

# task 1-1
# input_shapes = [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18), (1, 250, 12), (1, 20, 10), (1, 20), (1, 20)]
# output_shapes = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]


# input_shapes = [(1, 25, 15), (1, 25, 40), (1, 25, 18), (1, 25, 18), (1, 250, 12), (1, 20, 10), (1, 20), (1, 20)]
# output_shapes = [(1, 1, 170), (1, 1, 170)]
# output_shapes = [(1, 2, 170), (1, 2, 170)]

model = MultiBranchTimeCNNModel(input_shapes, output_shapes, D)

input = ([torch.randn((B,) + shape) for shape in input_shapes])
output_wanted = ([torch.randn((B,) + shape) for shape in output_shapes])

print( "\nINPUT SHAPES: ", [ arr.shape for arr in input ] )
print( "\nOUTPUT SHAPES: ", [ arr.shape for arr in model(*input) ] )
print( "\nOUTPUT SHAPES WANTED: ", [ arr.shape for arr in output_wanted ] )

# from torchinfo import summary

# shape_input = ([(B,) + shape for shape in input_shapes])
# summary(model, input_size=(shape_input))

