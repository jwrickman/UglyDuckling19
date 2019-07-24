import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from PIL import Image

from models.models import ConvAutoencoder
from dataset_prep.pipeline import get_dataset
from dataset_prep.make_records import create_tf_record


os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Training on only 1 GPU


# GLOBAL VARIABLES #
INPUT_PATH = '/media/storage4/molemap/dumped_images/'
SPLIT = 0.8  # Percentage of dataset to use for training vs validation
SEED = np.random.randint(10000)
out_path = "dataset_prep/z_records"
BATCH_SIZE = 50
gamma = 0.2
EPOCHS = 100


def visualize_img(img, name, destination):
    # Prepare output location
    out_path = Path(destination)
    if not out_path.exists():
        out_path.mkdir()
    img = np.round(img * 255).astype(np.uint8)
    fpath = out_path / "{}.png".format(name)
    print(str(fpath))
    im = Image.fromarray(img)
    im.save(str(fpath))


dataset, dataset_length = get_dataset(
    INPUT_PATH,
    SPLIT,
    EPOCHS,
    include_z_array=False,
    sort_by_patient=False,
    flags=['b', 'm'],
    flag_pos_1=-10,
    batch_size=BATCH_SIZE,
    seed=SEED
)

p_dataset, p_dataset_length = get_dataset(
    INPUT_PATH,
    SPLIT,
    EPOCHS,
    include_z_array=False,
    sort_by_patient=True,
    flags=['b', 'm'],
    flag_pos_1=64,
    flag_pos_2=67,
    aug_train=False
)

im, label, full = dataset.make_one_shot_iterator().get_next()

fim, flabel, orig_img = p_dataset.make_one_shot_iterator().get_next()

train_steps = dataset_length[0] // BATCH_SIZE
val_steps = dataset_length[1] // BATCH_SIZE


cae = ConvAutoencoder()

reconstruction, prediction = cae(im, training=False)


recon_loss = tf.keras.losses.MeanSquaredError()(im, reconstruction)

pred_loss = tf.keras.losses.BinaryCrossentropy()(label, prediction)

total_loss = recon_loss + (gamma * pred_loss)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(recon_loss)

init_op = tf.initializers.global_variables()


z_vector = cae.get_z_scores(fim)
frec, fpred = cae(fim, training=False)


with tf.Session() as sess:
    sess.run(init_op)

    for epoch in range(EPOCHS):
        ep_rec_loss_t = 0.0
        ep_pre_loss_t = 0.0
        ep_tot_loss_t = 0.0
        ep_rec_loss_v = 0.0
        ep_pre_loss_v = 0.0
        ep_tot_loss_v = 0.0

        for step in range(train_steps):
            _, b_rec_l, b_pre_l, b_tot_l = sess.run([train_op,
                                                     recon_loss,
                                                     pred_loss,
                                                     total_loss
                                                     ])

            ep_rec_loss_t += b_rec_l
            ep_pre_loss_t += b_pre_l
            ep_tot_loss_t += b_tot_l
        for step in range(val_steps):
            b_rec_l, b_pre_l, b_tot_l = sess.run([recon_loss,
                                                  pred_loss,
                                                  total_loss
                                                  ])
            ep_rec_loss_v += b_rec_l
            ep_pre_loss_v += b_pre_l
            ep_tot_loss_v += b_tot_l

        print("EPOCH: {}\nTRAIN: Tot: {}, Rec: {}, Pred: {}\n".format(
                epoch,
                ep_tot_loss_t / train_steps,
                ep_rec_loss_t / train_steps,
                ep_pre_loss_t / train_steps) +

              "VAL:   Tot: {}, Rec: {}, Pred: {}".format(
                ep_tot_loss_v / val_steps,
                ep_rec_loss_v / val_steps,
                ep_pre_loss_v / val_steps))

    print("CAE Training Complete")
    print("Making Patient Records")
    count = 0
    for pid in range(106 + 1):
        forig_img, label, z, rec = sess.run([orig_img, flabel, z_vector, frec])
        for i in range(label.shape[0]):
            if i < 5:
                visualize_img(
                    rec[i],
                    'recon{:03}_{:03}_{}_{:04}'.format(
                        pid, i, int(label[i][1]), count),
                    './visualize_reconstructions'
                )
            filename = "case{:03}_{:03}_{}_{:04}.tfrecord".format(
                pid, i, int(label[i][1]), count)
            count += 1
            example = create_tf_record(
                forig_img[i], label[i], z[i], out_path, filename, True)
