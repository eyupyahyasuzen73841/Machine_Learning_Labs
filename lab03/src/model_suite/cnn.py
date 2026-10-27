import torch
import torch.nn as nn

class EyupLeNetVariant(nn.Module):
    """
    Classical LeNet-inspired Convolutional Vision Classifier
    Topology: Conv(1->16, k=5) -> AvgPool(2x2) -> Conv(16->32, k=5) -> AvgPool(2x2) -> FC(512->120->10)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, num_classes)
        )

    def forward(self, x):
        return self.fc_head(self.conv_stack(x))
