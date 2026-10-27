import torch
import torch.nn as nn

class EyupMLPClassifier(nn.Module):
    """
    3-Layer LeakyReLU Multilayer Perceptron
    Topology: 784 -> 300 -> 100 -> 10
    """
    def __init__(self, in_features=784, hidden1=300, hidden2=100, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.encoder(x)
