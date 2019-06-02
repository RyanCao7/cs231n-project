from datetime import datetime

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
# ATTACKS = ['FGSM', 'RAND_FGSM', 'CW']
ATTACKS = ['FGSM', 'RAND_FGSM'] # DEBUGGING ONLY! TODO: REMOVE
# ATTACKS = ['CW'] # DEBUGGING ONLY! TODO: REMOVE

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
    'adversarial_train',
    'alpha',
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
    'adversarial_train',
    'alpha',
]

# torch.nn.[loss]
CLASSIFIER_CRITERIA = [
    'CrossEntropy',
]

# torch.nn.[loss]
GENERATOR_CRITERIA = [
    'ReconKLD',
]

# MNIST mean and std pulled from 
# https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
# CIFAR-10 mean and std pulled from 
# https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151 (see first comment)
# with reference to
# https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/dataloader.py#L16
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
FASHIONMNIST_MEAN = 0.2860407
FASHIONMNIST_STD = 0.35302424

# Constants for Box Sync
CONFIG_FILE = '.box_config.cfg'
CS231N_PROJECT_FOLDER = 'CS231n'
MODEL_FOLDER = 'models'
SYNC_DIRECTORIES = ['models', 'visuals']


# Standard datetime format
def get_cur_time():
    return datetime.now().strftime("%m-%d|%H:%M:%S")
