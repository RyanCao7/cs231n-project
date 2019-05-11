import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_Baseline(nn.Module):
    '''
    An implementation of the basic CNN architecture
    Madry et al used for training a model robust to
    adversarial attacks on MNIST.
    '''

    def __init__(self):
        super().__init__()

        # Initializing the conv layers with Kaiming and zero bias
        self.conv1 = nn.Conv2d(1, 32, 5, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant(self.conv1.bias, 0)

        self.conv2 = nn.Conv2d(32, 64, 5, bias=True)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant(self.conv2.bias, 0)

        # Initializing the fully connected layer
        self.fc = nn.Linear(1024, 10, bias=True)
        nn.inint.kaiming_normal_(self.fc.weight)
        nn.init.constant(self.fc.bias, 0)

    def forward(self, x):
        hidden_1 = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=2)
        hidden_2 = F.max_pool2d(F.relu(self.conv2(hidden_1)), 2, stride=2)

        N = hidden_2.shape[0]
        hidden_2 = hidden_2.view(N, -1)

        return self.fc(hidden_2)
