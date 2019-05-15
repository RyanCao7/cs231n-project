# TODO: Add VAE and other architectures to this list
MODEL_ARCHITECTURE_NAMES = [
    'Classifier_A', 
    'Classifier_B', 
    'Classifier_C', 
    'Classifier_D', 
    'Classifier_E',
    'VAE'
]

# Powers of 2.
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]

# torchvision.datasets.[dataset]
DATASETS = ['MNIST', 'CIFAR-10', 'Fashion-MNIST']

# torch.nn.[loss]
CRITERIA = ['CrossEntropyLoss'] # TODO: Add more criteria

# torch.optim
OPTIMIZERS = ['SGD', 'Adam', 'RMSProp']

# For state editing
EDITABLE_STATE_VARS = [
    'total_epochs',
    'learning_rate',
    'momentum',
    'weight_decay',
    'print_frequency',
    'save_every',
    'evaluate',
]

# For state setup
SETUP_STATE_VARS = [
    'batch_size',
    'dataset',
    'total_epochs',
    'learning_rate',
    'momentum',
    'weight_decay',
    'print_frequency',
    'save_every',
    'evaluate',
    'seed',
]