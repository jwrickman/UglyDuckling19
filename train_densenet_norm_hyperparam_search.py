import os
import tensorflow as tf
from dataset_prep.pipeline import get_dataset
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# GLOBAL VARIABLES #
INPUT_PATH = 'dataset_prep/z_records/'
BATCH_SIZE = 50  # Batch size
EPOCHS = 30 # Number of Epochs to train model for
SPLIT = 0.8  # Percentage of dataset to use for training vs validation
AUG_TRAIN = 'full'  # If True, dataset will be augmented (Flipping, noise, etc)
AUG_VAL = 'none'
SEED = 2222
EPOCHS_PER_CKPT = 50
USE_Z = False

if USE_Z:
    name = 'ugly_{}'.format(SEED)
else:
    name = 'norm_{}'.format(SEED)


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

img, z_array, label = dataset.make_one_shot_iterator().get_next()

p = {'lr1': [0.001, 0.01, 0.005],
     'lr2': [0.001, 0.01, 0.005],
     'dropout': [0.1, 0.3, 0.5, 0.6],
     'unfreeze': (0, 20, 4),
     'freeze_classifier': [True, False],
     }




def densenet_norm(x_train, y_train, x_val, y_val, params):


    base_model = tf.keras.applications.DenseNet121(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet',
                                                pooling='avg')

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dropout(params['dropout']),
        tf.keras.layers.Dense(
            2,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(params['regularizer']))
        ])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_acc',
        verbose=1,
        mode='auto',
        patience=2,
        restore_best_weights=True
    )

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=params['lr1']),
        metrics=['acc']
    )

    initial_train = model.fit(
            x=x_train,
            y=y_train,
            epochs=30,
            steps_per_epoch=num_train_steps,
            validation_data=(x_val, y_val),
            validation_steps=num_val_steps,
            callbacks=[early_stop]
    )

    for layer in model.layers[len(base_model.layers)-params['unfreeze']:
                              len(base_model.layers)-1]:
        layer.trainable = True

    if params['freeze_classifier']:
        for layer in model.layers[len(base_model.layers)-1:]:
            layer.trainable = False

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=params['lr2']),
        metrics=['acc']
    )

    fine_tune = model.fit(
            x=x_train,
            y=y_train,
            epochs=50,
            initial_epoch=len(initial_train.epoch)+2,
            steps_per_epoch=num_train_steps,
            validation_data=(x_val, y_val),
            validation_steps=num_val_steps,
            callbacks=[early_stop]
    )

    return model, fine_tune


def scan(x, y, train_fn, grid_downsample=0.01, dataset_name, experiment_no):



t = ta.Scan(x=img,
            y=label,
            model=densenet_norm,
            grid_downsample=0.01,
            params=p,
            dataset_name='molemap',
            experiment_no='1')
