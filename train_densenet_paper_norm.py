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


img_dim = (224, 224, 3)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.4


base_model = tf.keras.applications.DenseNet121(input_shape=img_dim,
                                               include_top=False,
                                               weights='imagenet',
                                               pooling='avg')


img_input = tf.keras.Input(shape=(224, 224, 3))
z_input = tf.keras.Input(shape=(784,))

z = tf.keras.layers.Dropout(rate=dropout_rate)(z_input)
z = tf.keras.layers.Dense(1,
                          activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.3))(z)
z_model = tf.keras.Model(inputs=z_input, outputs=z)
combined = tf.keras.layers.concatenate([base_model.output, z_model.output])
x = tf.keras.layers.Dropout(rate=dropout_rate)(combined)







base_model.trainable=False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(
        2,
        activation='softmax',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_acc',
    verbose=1,
    mode='auto',
    patience=4,
    restore_best_weights=True
)

filepath = "models/saved_models/" + name + "_best.ckpt"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    monitor='val_acc',
    save_best_only=True
)



base_layers = len(base_model.layers)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


img, z_array, label = dataset.make_one_shot_iterator().get_next()

history1 = model.fit(
        x=img,
        y=label,
        epochs=2,
        steps_per_epoch=num_train_steps,
        validation_data=(img, label),
        validation_steps=num_val_steps,
        callbacks=[checkpoint])

weights = model.layers[-1].get_weights()

x = tf.keras.layers.Dropout(rate=dropout_rate)(base_model.output)

predictions = tf.keras.layers.Dense(
    2,
    activation='softmax',
    kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
model = tf.keras.Model(
    inputs=base_model.input, outputs=predictions)

model.layers[-1].set_weights(weights)

for layer in model.layers[:len(base_model.layers)-1]:
    layer.trainable=False

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history2 = model.fit(
    x=img,
    y=label,
    epochs=30,
    initial_epoch=2,
    steps_per_epoch=num_train_steps,
    validation_data=(img, label),
    validation_steps=num_val_steps,
    callbacks=[early_stop, checkpoint]
)


print("Unfreezing last 10 layers of densenet")


i = 0
# Unfreezing last 10 layers of DenseNet121 for fine-tuning
for layer in model.layers[len(base_model.layers)-11:len(base_model.layers)-1]:
    layer.trainable = True

for layer in model.layers[len(base_model.layers)-1:]:
    layer.trainable = False


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    x=img,
    y=label,
    epochs=30,
    initial_epoch=len(history2.epoch)+2,
    steps_per_epoch=num_train_steps,
    validation_data=(img, label),
    validation_steps=num_val_steps,
    callbacks=[early_stop, checkpoint])

print(history2)

"""
weights = model.layers[-1].get_weights()
weights = [np.append(weights[0], np.array([[-0.001, 0.001]]), axis=0), weights[1]]

predictions = tf.keras.layers.Dense(
    2,
    activation='softmax',
    kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
model = tf.keras.Model(
    inputs=[base_model.input, z_model.input], outputs=predictions)

model.layers[-1].set_weights(weights)

for layer in model.layers[:len(base_model.layers)-1]:
    layer.trainable=False
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])



history = model.fit(
    x=[img, z_array],
    y=label,
    epochs=19,
    steps_per_epoch=num_train_steps,
    validation_data=([img, z_array], label),
    validation_steps=num_val_steps
)

if USE_Z:
    history = model.fit(
        x=[img, z_array],
        y=label,
        epochs=EPOCHS,
        steps_per_epoch=num_train_steps,
        validation_data=([img, z_array], label),
        validation_steps=num_val_steps
    )
else:
    history = model.fit(
        x=img,
        y=label,
        epochs=EPOCHS,
        steps_per_epoch=num_train_steps,
        validation_data=(img, label),
        validation_steps=num_val_steps
    )


weights = model.layers[-1].get_weights()
print(weights[0][-1])
print(weights[0][1023])
print(weights[0][1022])

"""

