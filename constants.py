# All classifier models
CLASSIFIERS = [
    'Classifier_A', 
    'Classifier_B', 
    'Classifier_C', 
    'Classifier_D', 
    'Classifier_E',
]

# All VAE implementations
GENERATORS = [
    'VANILLA_VAE',
    'DEFENSE_VAE',
]

# Powers of 2.
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]

# Adversarial attack methods
ATTACKS = ['FGSM', 'RAND_FGSM', 'CW']

# torchvision.datasets.[dataset]
DATASETS = ['MNIST', 'CIFAR-10', 'Fashion-MNIST']

# torch.optim
OPTIMIZERS = ['SGD', 'Adam'] # , 'RMSProp']

# For state editing
EDITABLE_STATE_VARS = [
    'total_epochs',
    'learning_rate',
    'momentum',
    'weight_decay',
    'print_frequency',
    'save_every',
    'evaluate',
    'total_epochs',
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
    'optimizer',
    'criterion',
    'adversarial_train'
]

# torch.nn.[loss]
CLASSIFIER_CRITERIA = [
    'CrossEntropy',
]

# torch.nn.[loss]
GENERATOR_CRITERIA = [
    'ReconKLD',
]
