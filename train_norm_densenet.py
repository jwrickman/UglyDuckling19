from dataset_prep.pipeline import get_dataset
from train import run_train_sess
from models.densenet_isic import DenseNet
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# GLOBAL VARIABLES #
INPUT_PATH = 'dataset_prep/z_records/'
BATCH_SIZE = 50  # Batch size
EPOCHS = 100  # Number of Epochs to train model for
SPLIT = 0.8  # Percentage of dataset to use for training vs validation
AUG_TRAIN = 'full'  # If True, dataset will be augmented (Flipping, noise, etc)
AUG_VAL = 'minimal'
SEED = 3333
EPOCHS_PER_CKPT = 50
USE_Z = False

if USE_Z:
    name = 'ugly_{}'.format(SEED)
else:
    name = 'norm_{}'.format(SEED)

# Hyperparameter
growth_k = 24
nb_block = 3  # how many (dense block + Transition Layer) ?
dropout_rate = 0.4


dataset, dataset_length = get_dataset(
    INPUT_PATH,
    SPLIT,
    EPOCHS,
    include_z_array=True,
    sort_by_patient=False,
    flags=['0', '1'],
    flag_pos_1=-15,
    batch_size=BATCH_SIZE,
    seed=SEED,
    aug_train=AUG_TRAIN,
    aug_val=AUG_VAL,
    rebalance=True
)

num_train_steps = dataset_length[0] // BATCH_SIZE
num_val_steps = dataset_length[1] // BATCH_SIZE
print("Train_steps: {}, Val_steps:{}".format(num_train_steps, num_val_steps))

model = DenseNet(
    nb_blocks=nb_block,
    filters=growth_k,
    dropout=dropout_rate,
    use_z=USE_Z
)

run_train_sess(
    dataset,
    model,
    EPOCHS,
    EPOCHS_PER_CKPT,
    num_train_steps,
    num_val_steps,
    name
)
