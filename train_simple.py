import tensorflow as tf
from dataset_prep.pipeline import get_dataset
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# GLOBAL VARIABLES #
INPUT_PATH = 'dataset_prep/z_records/'
BATCH_SIZE = 50  # Batch size
EPOCHS = 100  # Number of Epochs to train model for
SPLIT = 0.8  # Percentage of dataset to use for training vs validation
AUG_TRAIN = True  # If True, dataset will be augmented (Flipping, noise, etc)
AUG_VAL = False
SEED = np.random.randint(10000)
EPOCHS_PER_CKPT = 50
init_learning_rate = 1e-4


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
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
training_flag = tf.placeholder(tf.bool)


x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(img)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(x)
x = tf.keras.layers.Flatten()(x)
z = tf.keras.layers.Dense(1, activation='relu')(z_array)
x = tf.concat([x, z], 1)
logits = tf.keras.layers.Dense(2, activation='softmax')(x)


cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

optimizer = tf.train.AdamOptimizer(
    learning_rate=learning_rate,
    epsilon=1e-4
)

train = optimizer.minimize(cost)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_ph')
acc_ph = tf.placeholder(tf.float32, shape=None, name='acc_ph')


loss_sm = tf.summary.scalar('loss', loss_ph)
acc_sm = tf.summary.scalar('accuracy', acc_ph)

performance_summaries = tf.summary.merge([loss_sm,
                                          acc_sm])

saver = tf.train.Saver(tf.global_variables())

log_dir = "logs/ugly" + str(SEED)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./ugly_model_{}'.format(SEED))
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    val_writer = tf.summary.FileWriter(log_dir + "/val")

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, EPOCHS + 1):
        if epoch == (EPOCHS * 0.5) or epoch == (EPOCHS * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        train_acc = 0.0
        train_loss = 0.0
        train_hist = np.zeros(2)
        val_acc = 0.0
        val_loss = 0.0
        val_hist = np.zeros(2)

        for step in range(num_train_steps):
            train_feed_dict = {
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, batch_loss, batch_acc, batch_labels = sess.run(
                [train, cost, accuracy, label],
                feed_dict=train_feed_dict
            )

            train_hist += (np.sum(batch_labels, axis=0) / BATCH_SIZE)
            train_loss += batch_loss
            train_acc += batch_acc

        for step in range(num_val_steps):
            val_feed_dict = {
                learning_rate: epoch_learning_rate,
                training_flag: True
            }
            batch_loss, batch_acc, batch_labels = sess.run(
                [cost, accuracy, label],
                feed_dict=val_feed_dict)
            val_loss += batch_loss
            val_acc += batch_acc
            val_hist += (np.sum(batch_labels, axis=0) / BATCH_SIZE)


        train_loss /= num_train_steps
        train_acc /= num_train_steps
        train_hist /= num_train_steps

        val_loss /= num_val_steps
        val_acc /= num_val_steps
        val_hist /= num_val_steps

        tr_sum = sess.run(
            performance_summaries,
            feed_dict={loss_ph: train_loss,
                       acc_ph: train_acc}
        )

        val_sum = sess.run(
            performance_summaries,
            feed_dict={loss_ph: val_loss,
                       acc_ph: val_acc}
        )

        train_writer.add_summary(tr_sum, epoch)
        val_writer.add_summary(val_sum, epoch)

        line = "UGLY: EPOCH: {}/{}, train_loss: {:.4f}, train_acc: {:.4f},"\
            .format(epoch, EPOCHS, train_loss, train_acc)
        line += " test_loss: {:.4f}, test_acc: {:.4f}".format(
            val_loss, val_acc)
        print(line)
        print(train_hist)
        print(val_hist)
        print()
        if (epoch + 1) % EPOCHS_PER_CKPT == 0:
            saver.save(
                sess=sess,
                save_path='./ugly_model_{}/dense.ckpt'.format(SEED)
            )
    train_writer.close()
    val_writer.close()
