import tensorflow as tf
from dataset_prep.pipeline import get_dataset
from models.densenet import DenseNet
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# GLOBAL VARIABLES #
INPUT_PATH = 'dataset_prep/z_records/'
BATCH_SIZE = 50  # Batch size
EPOCHS = 100  # Number of Epochs to train model for
SPLIT = 0.8  # Percentage of dataset to use for training vs validation
AUG_TRAIN = 'full'  # If True, dataset will be augmented (Flipping, noise, etc)
AUG_VAL = 'minimal'
SEED = np.random.randint(10000)
EPOCHS_PER_CKPT = 50

# Hyperparameter
growth_k = 24
nb_block = 3  # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-4  # AdamOptimizer epsilon
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
img, label, z_array = dataset.make_one_shot_iterator().get_next()


model = DenseNet(
    x=img,
    nb_blocks=nb_block,
    filters=growth_k,
    dropout_rate=dropout_rate
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics='accuracy'
)

history = model.fit(
    x=[img, z_array],
    y=label,
    steps_per_epoch=num_train_steps,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_steps=num_val_steps
)
