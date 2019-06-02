import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import data_utils
import constants


def initialize_model(model):
    '''
    Performs Kaiming weight initialzation for a 
    given model.

    Keyword arguments:
    > model (nn.Module) -- The model whose weights are
        to be initialized.
    '''
    model.apply(initialize_weight)


def initialize_weight(param):
    '''
    Performs Kaiming weight initialization for a
    given parameter.
    
    Keyword arguments:
    > param (torch.nn.Module) -- the parameter whose 
        weights are to be initialized.
    '''
    if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear) or isinstance(param, nn.ConvTranspose2d):
        init.kaiming_normal_(param.weight, mode='fan_in', nonlinearity='relu')
        init.constant_(param.bias, 0)

            
def normalize(data, dataset_name='Fashion-MNIST'):
    '''
    Returns normalized version of data, as specified by
    dataset_name.
    
    Keyword arguments:
    > dataset_name (string) -- Name of the dataset being
        trained on. Must be one of 'MNIST', 'CIFAR-10', or
        'Fashion-MNIST'.
    > data (torch.tensor) -- The data to be normalized.
    '''
    if dataset_name == 'MNIST':
        mean, std = constants.MNIST_MEAN, constants.MNIST_STD
    elif dataset_name == 'CIFAR-10':
        mean, std = constants.CIFAR10_MEAN, constants.CIFAR10_STD
    elif dataset_name == 'Fashion-MNIST':
        mean, std = constants.FASHIONMNIST_MEAN, constants.FASHIONMNIST_STD
    else:
        raise Exception('Error: dataset_name must be one of {\'MNIST\', \'CIFAR-10\', '
                        '\'Fashion-MINST\'}.')
    return (data - mean) / std


def denormalize(normalized_data, dataset_name='Fashion-MNIST'):
    '''
    Returns denormalized version of normalized data, as specified
    by dataset_name.
    
    Keyword arguments:
    > dataset_name (string) -- Name of the dataset being
        trained on. Must be one of 'MNIST', 'CIFAR-10', or
        'Fashion-MNIST'.
    > normalized_data (torch.tensor) -- The data to be denormalized.
    '''
    if dataset_name == 'MNIST':
        mean, std = constants.MNIST_MEAN, constants.MNIST_STD
    elif dataset_name == 'CIFAR-10':
        mean, std = constants.CIFAR10_MEAN, constants.CIFAR10_STD
    elif dataset_name == 'Fashion-MNIST':
        mean, std = constants.FASHIONMNIST_MEAN, constants.FASHIONMNIST_STD
    else:
        raise Exception('Error: dataset_name must be one of {\'MNIST\', \'CIFAR-10\', '
                        '\'Fashion-MINST\'}.')
    return (data * std) + mean

    
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
        
        # Normalize first (WARNING: ASSUMES FASHION-MNIST)
        x = normalize(x)
        
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
        # Normalize first (WARNING: ASSUMES FASHION-MNIST)
        x = normalize(x)
        
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
        # Normalize first (WARNING: ASSUMES FASHION-MNIST)
        x = normalize(x)
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
        # Normalize first (WARNING: ASSUMES FASHION-MNIST)
        x = normalize(x)
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
        # Normalize first (WARNING: ASSUMES FASHION-MNIST)
        x = normalize(x)
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

    
class Defense_VAE(nn.Module):
    '''
    An implementation of the Defense-VAE architecture presented in
    "Defense-VAE: A Fast and Accurate Defense against Adversarial Attacks"
    by Li and Ji.
    '''

    def __init__(self):
        super().__init__()
        
        # Encoder parameters
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2, padding=3, bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc11 = nn.Linear(4096, 128, bias=True)
        self.fc12 = nn.Linear(4096, 128, bias=True)

        # Decoder parameters
        self.fc2 = nn.Linear(128, 4096, bias=True)

        self.tconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True)
        self.tbn1 = nn.BatchNorm2d(128)

        self.tconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True)
        self.tbn2 = nn.BatchNorm2d(64)

        self.tconv3 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=3, bias=True)
        self.tbn3 = nn.BatchNorm2d(64)

        # I removed the BN + ReLU of the final layer to use sigmoid instead.
        self.tconv4 = nn.ConvTranspose2d(64, 1, 5, padding=2, bias=True)

    def encode(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        h4 = F.relu(self.bn4(self.conv4(h3)))

        # Flatten
        N = h4.shape[0]
        h4 = h4.view(N, -1)

        return self.fc11(h4), self.fc12(h4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h1 = F.relu(self.fc2(z))
        h1 = h1.view(-1, 256, 4, 4)

        h2 = F.relu(self.tbn1(self.tconv1(h1)))
        h3 = F.relu(self.tbn2(self.tconv2(h2)))
        h4 = F.relu(self.tbn3(self.tconv3(h3)))

        out = torch.sigmoid(self.tconv4(h4))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
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
        reconv_x = reconv_x.reshape(-1, 1, 28, 28) # Unflattening for conv net
        return self.classifier(reconv_x)
        
    # Deprecated
    def normalize(self, data):
        return (data - torch.mean(data, dim=0, keepdim=True)) / torch.std(data, dim=0, keepdim=True)
        
        
