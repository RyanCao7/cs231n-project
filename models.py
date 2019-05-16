import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# TODO: Implement Kaiming initialization for model weights here!
def initialize_model(model):
    for name, param in model.named_parameters():
        if name.endswith('.weight'):
            init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        elif name.endswith('.bias'):
            init.constant_(param, 0)

            
class Classifier_A(nn.Module):
    '''
    An implementation of Classifier A used by Samangouei et al in their paper 
    "Defense-GANL Protecting Classifiers Against Adversarial Attacks Using 
    Generative Models."

    Weight initialization should be performed after initializing the model.
    '''

    def __init__(self):
        super().__init__()

        # Initializing the conv layers
        self.conv_1 = nn.Conv2d(1, 64, 5, padding=2, bias=True)
        self.conv_2 = nn.Conv2d(64, 64, 5, stride=2, bias=True)

        # Initializing the fully connected layer
        self.drop_1 = nn.Dropout(p=0.25)
        self.fc_1 = nn.Linear(9216, 128, bias=True)

        self.drop_2 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        # Forward prop through conv layers
        hidden_1 = F.relu(self.conv_1(x))
        hidden_2 = F.relu(self.conv_2(hidden_1))

        # Flatten image and apply dropout
        N = hidden_2.shape[0]
        hidden_2 = hidden_2.view(N, -1)
        hidden_2 = self.drop_1(hidden_2)

        # FC layers
        hidden_3 = self.drop_2(F.relu(self.fc_1(hidden_2)))
        out = self.fc_2(hidden_3)

        return out

    
class Classifier_B(nn.Module):
    '''
    An implementation of Classifier B used by Samangouei et al in their paper 
    "Defense-GANL Protecting Classifiers Against Adversarial Attacks Using 
    Generative Models."

    Weight initialization should be performed after initializing the model.
    '''
    
    def __init__(self):
        super().__init__()

        self.drop_1 = nn.Dropout(p=0.2)
        self.conv_1 = nn.Conv2d(1, 64, 8, stride=2, padding=5, bias=True)

        self.conv_2 = nn.Conv2d(64, 128, 6, stride=2, bias=True)

        self.conv_3 = nn.Conv2d(128, 128, 5, bias=True)
        self.drop_2 = nn.Dropout(p=0.5)

        self.fc = nn.Linear(512, 10, bias=True)

    def forward(self, x):
        x = self.drop_1(x)
        hidden_1 = F.relu(self.conv_1(x))
        hidden_2 = F.relu(self.conv_2(hidden_1))        
        hidden_3 = F.relu(self.conv_3(hidden_2))

        # Flatten
        N = hidden_3.shape[0]
        hidden_3 = hidden_3.view(N, -1)
        hidden_3 = self.drop_2(hidden_3)

        out = self.fc(hidden_3)
        return out

    
class Classifier_C(nn.Module):
    '''
    An implementation of Classifier C used by Samangouei et al in their paper 
    "Defense-GANL Protecting Classifiers Against Adversarial Attacks Using
    Generative Models."

    Weight initialization should be performed after initializing the model.
    '''

    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 128, 3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(128, 64, 3, stride=2, bias=True)

        self.drop_1 = nn.Dropout(p=0.25)
        self.fc_1 = nn.Linear(10816, 128, bias=True)
        self.drop_2 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        hidden_1 = F.relu(self.conv_1(x))
        hidden_2 = F.relu(self.conv_2(hidden_1))

        N = hidden_2.shape[0]
        hidden_2 = hidden_2.view(N, -1)
        hidden_2 = self.drop_1(hidden_2)

        hidden_3 = self.drop_2(F.relu(self.fc_1(hidden_2)))
        out = self.fc_2(hidden_3)
        return out

    
class Classifier_D(nn.Module):
    '''
    An implementation of Classifier D used by Samangouei et al in their paper 
    "Defense-GANL Protecting Classifiers Against Adversarial Attacks Using 
    Generative Models."

    Weight initialization should be performed after initializing the model.
    '''
    
    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Linear(784, 200, bias=True)
        self.drop_1 = nn.Dropout(p=0.5)

        self.fc_2 = nn.Linear(200, 200, bias=True)
        self.drop_2 = nn.Dropout(p=0.5)

        self.fc_3 = nn.Linear(200, 10, bias=True)

    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, -1)

        hidden_1 = self.drop_1(F.relu(self.fc_1(x)))
        hidden_2 = self.drop_2(F.relu(self.fc_2(hidden_1)))
        out = self.fc_3(hidden_2)
        return out

    
class Classifier_E(nn.Module):
    '''
    An implementation of Classifier E used by Samangouei et al in their paper
    "Defense-GANL Protecting Classifiers Against Adversarial Attacks Using 
    Generative Models."

    Weight initialization should be performed after initializing the model.
    '''
    
    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Linear(784, 200, bias=True)
        self.fc_2 = nn.Linear(200, 200, bias=True)
        self.fc_3 = nn.Linear(200, 10, bias=True)
        
    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, -1)

        hidden_1 = F.relu(self.fc_1(x))
        hidden_2 = F.relu(self.fc_2(hidden_1))
        out = self.fc_3(hidden_2)
        return out

    
class VAE(nn.Module):
    '''
    An implementation of the paper "Stochastic Gradient VB and the Variational
    Auto-Encoder" by Kingma and Welling. Shamelessly ported and modified from the
    PyTorch example library.
    '''

    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return x, self.decode(z), mu, logvar

    
class DAVAE(nn.Module):
    '''
    Defense Against Adversarial Attacks VAE.
    Wrapper around a VAE + classifier.
    '''
    def __init__(self, classifier, generator):
        super(DAVAE, self).__init__()
        self.classifier = classifier
        self.generator = generator
        
    def forward(self, x):
        _, reconv_x, _, _ = self.generator(x)
        reconv_x = self.normalize(reconv_x)
        reconv_x = reconv_x.reshape(-1, 1, 28, 28) # Unflattening for conv net
        return self.classifier(reconv_x)
        
    def normalize(self, data):
        return (data - torch.mean(data, dim=0, keepdim=True)) / torch.std(data, dim=0, keepdim=True)
        
        