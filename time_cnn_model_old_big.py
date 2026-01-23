import torch
import torch.nn as nn
import numpy as np


padding = 1
kernel_size = 3
stride = 2
layers = 2 #not changing the model yet!!!!

def compute_compressed_size(input_size, layers, kernel_size, stride, padding, output_padding=0):
    """
    Computes the size of a dimension after multiple convolutional layers.

    Args:
        input_size (int): Original size of the dimension (e.g., time steps or height).
        layers (int): Number of layers.
        kernel_size (int): Kernel size of the convolution.
        stride (int): Stride of the convolution. Default is 1.
        padding (int): Padding added to both sides. Default is 0.
        output_padding (int): Output padding (used in transposed conv). Default is 0.

    Returns:
        int: Compressed size after all layers.
    """
    size = input_size
    for _ in range(layers):
        size = int(np.ceil((size + 2*padding - kernel_size - output_padding) / stride + 1))
    return size



# ======================================================================================================================
class Conv1DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D):
        super().__init__()

        self.D = D
        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]
        self.ts_comp = compute_compressed_size(self.ts_var, layers, kernel_size, stride, padding)

        self.cnn = nn.Sequential(
            nn.BatchNorm1d(self.n_var),
            nn.Conv1d(self.n_var, D, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=padding),
            nn.BatchNorm1d(D),
            nn.Conv1d(D, 2 * D, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(2, padding=padding)
        )
        self.fc = nn.Linear(2 * D * self.ts_comp, D )
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # print('\nConv1D')
        # print('in', x.shape)
        x = self.cnn(x)
        # print('after cnn', x.shape)
        x = x.flatten(start_dim=1)
        # print('after flatten', x.shape)
        x = self.fc(x)
        # print('out after fc', x.shape)   

        return x
    
# ======================================================================================================================
class Conv1DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D):
        super().__init__()

        self.D = D
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]
        self.ts_comp = compute_compressed_size(self.ts_var, layers, kernel_size, stride, padding)

        self.fc = nn.Linear(D, 2 * D * self.ts_comp )

        self.transposecnn = nn.Sequential(
            nn.ConvTranspose1d(2 * D, D, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose1d(D, self.n_var, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.ReLU(),
        )

    def forward(self, x):
        # # print('\nTransConv1D')
        # print(x.shape)     
        x = self.fc(x)  
        # print(x.shape)
        x = x.view(-1, 2*self.D, self.ts_comp)  
        # print(x.shape) 
        x = self.transposecnn(x)
        # print(x.shape)
        x = x[:, :, :self.ts_var]
        # print(x.shape)     

        return x



# ======================================================================================================================
class Conv2DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D):
        super().__init__()

        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]
        self.height_var = input_shape[2]
        
        self.ts_comp = compute_compressed_size(self.ts_var, layers, kernel_size, stride, padding)
        self.height_comp = compute_compressed_size(self.height_var, layers, kernel_size, stride, padding)

        self.cnn = nn.Sequential(
            nn.BatchNorm2d(self.n_var),
            nn.Conv2d(self.n_var, D, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1),
            nn.BatchNorm2d(D),
            nn.Conv2d(D, 2 * D, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        # self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2 * D * self.ts_comp * self.height_comp, D )

    def forward(self, x):
        # print('\nConv2D')
        # print('in', x.shape)
        x = self.cnn(x)
        # print(x.shape)
        x = x.flatten(start_dim=1)
        # print('after flatten', x.shape)
        x = self.fc(x)
        # print('out after fc', x.shape)
        
        return x


# ======================================================================================================================
class Conv2DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D):
        super().__init__()

        self.D = D
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]
        self.height_var = output_shape[2]

        # determine time comp representation
        self.ts_comp = compute_compressed_size(self.ts_var, layers, kernel_size, stride, padding)
        # print('ts_comp 2D', self.ts_comp)

        # determine height comp representation
        self.height_comp = compute_compressed_size(self.height_var, layers, kernel_size, stride, padding)
        # print(self.height_comp)
        # print('height_comp 2D', self.height_comp)

        self.fc = nn.Linear(D, 2 * D * self.ts_comp * self.height_comp)

        self.transposecnn = nn.Sequential(
            nn.ConvTranspose2d(2 * D, D, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(D, self.n_var, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.ReLU(),
        )

    def forward(self, x):
        # print('\nTransConv2D')
        # print('in', x.shape)     
        x = self.fc(x)  
        # print('after fc', x.shape)
        x = x.view(-1, 2*self.D, self.ts_comp, self.height_comp)  
        # print('after reshape', x.shape) 
        x = self.transposecnn(x)
        # print('after transconv', x.shape)
        x = x[:, :, :self.ts_var, :self.height_var]
        # print('out', x.shape)     

        return x

# ======================================================================================================================
class Conv3DEncoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_shape, D):
        super().__init__()

        self.n_var = input_shape[0]
        self.ts_var = input_shape[1]
        self.height_var = input_shape[2]
        self.weight_var = input_shape[3]
        
        self.ts_comp = compute_compressed_size(self.ts_var, layers, kernel_size, stride, padding)
        self.height_comp = compute_compressed_size(self.height_var, layers, kernel_size, stride, padding)
        self.weight_comp = compute_compressed_size(self.weight_var, layers, kernel_size, stride, padding)
        
        self.cnn = nn.Sequential(
            nn.BatchNorm3d(self.n_var),
            nn.Conv3d(self.n_var, D, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool3d(2, padding=1),
            nn.BatchNorm3d(D),
            nn.Conv3d(D, 2 * D, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool3d(2, padding=1)
        )
        # self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2 * D * self.ts_comp * self.height_comp * self.weight_comp, D )


    def forward(self, x):
        # print('\nConv3D')
        # print('in', x.shape)
        x = self.cnn(x)
        # print('after cnn', x.shape)
        x = x.flatten(start_dim=1)
        # print('after flatten', x.shape)
        x = self.fc(x)
        # print('out after fc', x.shape)
        
        return x


# ======================================================================================================================
class Conv3DDecoder(nn.Module):

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_shape, D):
        super().__init__()

        self.D = D
        self.n_var = output_shape[0]
        self.ts_var = output_shape[1]
        self.height_var = output_shape[2]
        self.weight_var = output_shape[3]

        # determine time comp representation
        self.ts_comp = compute_compressed_size(self.ts_var, layers, kernel_size, stride, padding)
        # print('ts_comp 3D', self.ts_comp)

        # determine height comp representation
        self.height_comp = compute_compressed_size(self.height_var, layers, kernel_size, stride, padding)
        # print(self.height_comp)
        # print('height_comp 3D', self.height_comp)

        # determine weight comp representation
        self.weight_comp = compute_compressed_size(self.weight_var, layers, kernel_size, stride, padding)
        # print(self.weight_comp)
        # print('weight_comp 3D', self.weight_comp)

        self.fc = nn.Linear(D, 2 * D * self.ts_comp * self.height_comp * self.weight_comp)

        self.transposecnn = nn.Sequential(
            nn.ConvTranspose3d(2 * D, D, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose3d(D, self.n_var, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.ReLU(),
        )

    def forward(self, x):
        # print('\nTransConv3D')
        # print('in', x.shape)     
        x = self.fc(x)  
        # print('after fc', x.shape)
        x = x.view(-1, 2*self.D, self.ts_comp, self.height_comp, self.weight_comp)  
        # print('after reshape', x.shape) 
        x = self.transposecnn(x)
        # print('after transconv', x.shape)
        x = x[:, :, :self.ts_var, :self.height_var, :self.weight_var]
        # print('out', x.shape)     

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
                branch = Conv3DEncoder(shape, D)
            elif len(shape) == 3:  # e.g., (1, T, 15) profiles evolving in time
                branch = Conv2DEncoder(shape, D)
            elif len(shape) == 2:  # e.g., (7, T, ) time series evolving in time
                branch = Conv1DEncoder(shape, D)
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
            # print('shape', var_shape, 'len', len(var_shape))
            if len(var_shape) == 4:
                branch = Conv3DDecoder(var_shape, D)
            elif len(var_shape) == 3:
                branch = Conv2DDecoder(var_shape, D)
            elif len(var_shape) == 2:
                branch = Conv1DDecoder(var_shape, D)
            else:
                raise ValueError(f"Unsupported input shape: {var_shape}")
            self.output_branches.append(branch)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, *inputs):
        branch_outputs = []

        for branch, x in zip(self.input_branches, inputs):
            out = branch(x)
            branch_outputs.append(out)
        
        # print('\nCommon Layer to flatten time')     
        merged = torch.cat(branch_outputs, dim=1)
        # print(merged.shape)        
        merged = self.backbone(merged)
        # print('after backbone', merged.shape)   

        decoded_representation = []

        for branch in self.output_branches :
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

# # input_shapes = [(1, 7), (1, 1), (1, 65), (3, 12, 65), (1, 5, 6, 7), (1, 1, 1, 1)]
# input_shapes = [(1, 7), (3, 12, 65), (1, 50, 15, 6)]
# input = ([torch.randn((B,) + shape) for shape in input_shapes])

# output_shapes = [(1, 1), (1, 200, 5), (1, 50, 15, 6), (1, 200, 15, 65), (1, 1, 1, 1)]

# model = MultiBranchTimeCNNModel(input_shapes, output_shapes, D)

# print( "\nINPUT SHAPES: ", [ arr.shape for arr in input ] )
# print( "\nOUTPUT SHAPES: ", [ arr.shape for arr in model(*input) ] )

# print('stop')
# from torchinfo import summary
# summary(model, input_size=((700, 1, 7), (700, 3, 12, 65), (700, 1, 5, 6, 7), (700, 1, 1, 1, 1)))


input_shapes = [(1, 1, 120), (1, 1, 120), (1, 250, 3), (1, 250, 18), (1, 250, 18), (1, 220), (1, 220), (1, 220), (1, 220)]
output_shapes = [(1, 2500, 3), (1, 2500, 18), (1, 2500, 18)]

model = MultiBranchTimeCNNModel(input_shapes, output_shapes, D)

input = ([torch.randn((B,) + shape) for shape in input_shapes])
output_wanted = ([torch.randn((B,) + shape) for shape in output_shapes])

print( "\nINPUT SHAPES: ", [ arr.shape for arr in input ] )
print( "\nOUTPUT SHAPES: ", [ arr.shape for arr in model(*input) ] )
print( "\nOUTPUT SHAPES WANTED: ", [ arr.shape for arr in output_wanted ] )

from torchinfo import summary
shape_input = ([(B,) + shape for shape in input_shapes])
summary(model, input_size=(shape_input))

