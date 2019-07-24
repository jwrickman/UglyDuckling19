import tensorflow as tf
from pipeline import get_dataset
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# GLOBAL VARIABLES #
INPUT_PATH = '/media/storage4/molemap/dumped_images/'

SPLIT = 0.8  # Percentage of dataset to use for training vs validation
SEED = np.random.randint(10000)

EPOCHS = 1

pred_master_dataset, dataset_length = get_dataset(
    INPUT_PATH,
    SPLIT,
    EPOCHS,
    include_z_array=False,
    sort_by_patient=True,
    flags=['b', 'm'],
    flag_pos_1=64,
    flag_pos_2=67,
    batch_size=1,
    seed=SEED
)


im, label = pred_master_dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    im = sess.run(im)
    print(dataset_length)
