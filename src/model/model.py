# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SnoutNet(nn.Module):
    def __init__(self):
        super(SnoutNet, self).__init__()

        # Conv1: 64 filters, 3x3 kernel, stride 2, padding 2 -> output: 64 channels, 114x114
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool -> 57x57

        # Conv2: 128 filters, 3x3 kernel, stride 2, padding 2 -> output: 128 channels, 30x30
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool -> 15x15

        # Conv3: 256 filters, 3x3 kernel, stride 2, padding 2 -> output: 256 channels, 9x9
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool -> 4x4

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # FC1: input 4096 (4x4x256), output 1024
        self.fc2 = nn.Linear(1024, 1024)         # FC2: input 1024, output 1024
        self.fc3 = nn.Linear(1024, 2)            # FC3: input 1024, output 2 (for u, v coordinates)

    def forward(self, x):
        # Apply convolutions, ReLU, and pooling
        x = self.pool1(F.relu(self.conv1(x)))  # Conv1 -> Pool1, Output: [batch_size, 64, 57, 57]
        x = self.pool2(F.relu(self.conv2(x)))  # Conv2 -> Pool2, Output: [batch_size, 128, 15, 15]
        x = self.pool3(F.relu(self.conv3(x)))  # Conv3 -> Pool3, Output: [batch_size, 256, 4, 4]

        # Flatten for fully connected layers using reshape
        x = x.reshape(x.size(0), -1)  # Flatten to (batch_size, 4096)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))  # FC1: Output: [batch_size, 1024]
        x = F.relu(self.fc2(x))  # FC2: Output: [batch_size, 1024]
        return self.fc3(x)  # FC3: Output: [batch_size, 2] (u, v coordinates)
    
# Testing
# if __name__ == '__main__':
#     print("Testing architecture...")

#     # Get device
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = 'cuda'
#     print(f"Using device: {device}\n")

#     # Create a random dummy tensor with the specified shape
#     rand_tensor = torch.rand(16, 3, 227, 227)  # Batch size 16, RGB image, 227x227
#     print(f"Input: {rand_tensor.shape}")  # Display the shape of the tensor
#     rand_tensor = rand_tensor.to(device=device)

#     model = SnoutNet()
#     model.to(device)
#     model.eval()

#     output_tensor = model(rand_tensor)
#     print(f"Output: {output_tensor.shape}")  # Should output [16, 2]
