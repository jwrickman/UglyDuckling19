import os
import tensorflow as tf
from dataset_prep.pipeline import get_dataset
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# GLOBAL VARIABLES #
INPUT_PATH = 'dataset_prep/z_records/'
BATCH_SIZE = 50  # Batch size
EPOCHS = 30  # Number of Epochs to train model for
SPLIT = 0.8  # Percentage of dataset to use for training vs validation
AUG_TRAIN = 'full'  # If True, dataset will be augmented (Flipping, noise, etc)
AUG_VAL = 'minimal'
SEED = np.random.randint(10000)
EPOCHS_PER_CKPT = 50
USE_Z = True

if USE_Z:
    name = 'ugly_{}'.format(SEED)
else:
    name = 'norm_{}'.format(SEED)


# Dataset Prep #
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


img_dim = (224, 224, 3)
dropout_rate = 0.3


# Building Base DenseNet121 Model #
base_model = tf.keras.applications.DenseNet121(input_shape=img_dim,
                                               include_top=False,
                                               weights='imagenet',
                                               pooling='avg')

img_input = tf.keras.Input(shape=(224, 224, 3))

# Building IPCA UglyDuckling Model Part #
z_input = tf.keras.Input(shape=(784,))  # 784n vector ipca code

z = tf.keras.layers.Dense(
    256,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0))(z_input)

z = tf.keras.layers.Dropout(rate=dropout_rate / 2)(z)

z = tf.keras.layers.Dense(
    64,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0),
    activity_regularizer=tf.keras.regularizers.l2(0.001))(z)

z = tf.keras.layers.Dropout(rate=dropout_rate / 4)(z)

z = tf.keras.layers.Dense(
    1,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0))(z)

z_model = tf.keras.Model(inputs=z_input, outputs=z)

# Combining IPCA and DenseNet121 by appending IPCA node to output of DenseNet
combined = tf.keras.layers.concatenate([base_model.output, z_model.output])

combined = tf.keras.layers.Dropout(rate=dropout_rate)(combined)

base_model.trainable = False   # Freezing pretained imagenet weights

model = tf.keras.Sequential([  # Building custom classifier
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



model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


print("\nStarting Initial Training w/out IPCA Node\n")
initial_train = model.fit(  # Train classifier without IPCA node to stabilize
        x=img,
        y=label,
        epochs=2,
        steps_per_epoch=num_train_steps,
        validation_data=(img, label),
        validation_steps=num_val_steps,
        callbacks=[checkpoint])


# Get the weights of the custom classifier, and add weights for IPCA
weights = model.layers[-1].get_weights()
weights = [
    np.append(weights[0], np.array([[-0.001, 0.001]]), axis=0), weights[1]]

# Build model with IPCA node added
predictions = tf.keras.layers.Dense(
    2,
    activation='softmax',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)

model = tf.keras.Model(
    inputs=[base_model.input, z_model.input], outputs=predictions)


model.layers[-1].set_weights(weights)


# Freeze pretrained imagenet weights
for layer in model.layers[:len(base_model.layers)-1]:
    layer.trainable = False


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


print("\nStarting Training w/ IPCA Node\n")
ipca_train = model.fit(
    x=[img, z_array],
    y=label,
    epochs=30,
    initial_epoch=2,
    steps_per_epoch=num_train_steps,
    validation_data=([img, z_array], label),
    validation_steps=num_val_steps,
    callbacks=[early_stop, checkpoint]
)


# Unfreezing last 10 layers of DenseNet121 for fine-tuning
for layer in model.layers[len(base_model.layers)-11:len(base_model.layers)-1]:
    layer.trainable = True

for layer in model.layers[len(base_model.layers)-1:]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])


print("\nStarting Fine-Tuning\n")
fine_tune = model.fit(
    x=[img, z_array],
    y=label,
    epochs=30,
    initial_epoch=len(ipca_train.epoch)+2,
    steps_per_epoch=num_train_steps,
    validation_data=([img, z_array], label),
    validation_steps=num_val_steps,
    callbacks=[early_stop, checkpoint]
)



weights = model.layers[-1].get_weights()


see_model = tf.keras.Model(
    inputs=[base_model.input, z_model.input], outputs=model.layers[-2].output)


train_activations = see_model.predict([img, z_array], steps=num_train_steps)
activations = see_model.predict([img, z_array], steps=num_val_steps)

neg = activations * np.abs(np.transpose(weights[0])[0])
pos = activations * np.abs(np.transpose(weights[0])[1])


print("Average Neg Impact")
print("avg: {} std: {}".format(np.mean(neg), np.std(neg)))
print("ICPA Node Neg Impact")
print("avg: " + str(np.mean(neg, axis=0)[-1]) + " std: " + str(np.std(neg, axis=0)[-1]))

print("Average Pos Impact")
print("avg: " + str(np.mean(pos)) + " std: " + str(np.std(pos)))
print("ICPA Node Pos Impact")
print("avg: " + str(np.mean(pos, axis=0)[-1]) + " std: " + str(np.std(pos, axis=0)[-1]))
