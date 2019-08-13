import tensorflow as tf

def run_train_sess(dataset,
                   model,
                   epochs,
                   epochs_per_ckpt,
                   train_steps,
                   val_steps,
                   name):

    img, label, z_array = dataset.make_one_shot_iterator().get_next()

    training_flag = tf.placeholder(tf.bool)

    logits = model.call(img, z_array, training_flag)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
    )

    optimizer = tf.train.AdamOptimizer()

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

    log_dir = "logs/{}".format(name)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            './models/checkpoints/{}'.format(name)
        )
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
        val_writer = tf.summary.FileWriter(log_dir + "/val")

        for epoch in range(1, epochs + 1):
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            for step in range(train_steps):
                train_feed_dict = {
                    training_flag: True
                }

                _, batch_loss, batch_acc = sess.run(
                    [train, cost, accuracy],
                    feed_dict=train_feed_dict
                )

                train_loss += batch_loss
                train_acc += batch_acc

            for step in range(val_steps):
                val_feed_dict = {
                    training_flag: False
                }
                batch_loss, batch_acc = sess.run(
                    [cost, accuracy],
                    feed_dict=val_feed_dict)
                val_loss += batch_loss
                val_acc += batch_acc

            train_loss /= train_steps
            train_acc /= train_steps

            val_loss /= val_steps
            val_acc /= val_steps

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

            line = "{}: EPOCH: {}/{}, train_loss: {:.4f}, train_acc: {:.4f},"\
                .format(name, epoch, epochs, train_loss, train_acc)
            line += " test_loss: {:.4f}, test_acc: {:.4f} \n".format(
                val_loss, val_acc)
            print(line)
            if (epoch + 1) % epochs_per_ckpt == 0:
                saver.save(
                    sess=sess,
                    save_path='./models/checkpoints/{}/dense.ckpt'.format(name)
                )
        train_writer.close()
        val_writer.close()
