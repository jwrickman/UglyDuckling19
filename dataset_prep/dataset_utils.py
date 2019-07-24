import tensorflow as tf
import math

# Data augmentation hyper-parameters
_BRIGHTNESS_MAX_DELTA = 0.3
_CONTRAST_MAX_DELTA = 0.7


def _preprocess_image(image, is_training, new_size):
    """Pre-process a single image. Perform data augmentation if training.
    Parameters:
        image (np.array): [n x n x 3] image-array castable to tf.float32
        is_training (bool): whether to perform data augmentation or not
        new_size (int): image will be cropped to [new_size x new_size X 3]

    Returns:
        image (tf.tensor): [n,n,3] tf.tensor

    """
    image = tf.cast(image, tf.float32)/256.0
    if is_training.lower() == 'full':

        # Get random parameters of this augmentation
        angle = \
            tf.random_uniform((), minval=0, maxval=2*math.pi, dtype=tf.float32)
        dim = tf.cast(
            tf.random_normal((), mean=new_size+30.0, stddev=15.0), tf.int32
        )
        dim = tf.cond(tf.greater(dim, new_size), lambda: dim, lambda: new_size)

        # Perform augmentation operations
        image = tf.image.resize_images(image, (dim, dim))
        image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')
        image = tf.random_crop(
            image,
            [new_size, new_size, 3]
        )
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(
                image, max_delta=_BRIGHTNESS_MAX_DELTA
        )
        image = tf.image.random_contrast(
            image,
            lower=1.0-_CONTRAST_MAX_DELTA, upper=1.0+_CONTRAST_MAX_DELTA
        )
        return image

    elif is_training.lower() == 'minimal':
        image = tf.image.resize_images(image, (new_size+30, new_size+30))
        image = tf.random_crop(
            image,
            [new_size, new_size, 3]
        )
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        return image

    else:  # Perform no augmentations, just resize
        image = tf.image.resize_images(image, (new_size+30, new_size+30))
        image = tf.image.resize_image_with_crop_or_pad(
            image, new_size, new_size)
        return image


def _parse_record_orig(raw_record, is_training, new_size=64):
    """Parse a single instance record

    Parameters:
        raw_record (tf.data.example): a raw record from a tf.record
        is_training (bool): Whether to perfrom data augmentations on record
        new_size (int): Size to crop image to

    Returns:
    tuple(
        image (tf.tensor): a tf.tensor w/ shape [new_size,new_size,3]
        label (tf.tensor): a one-hot encoded tf.tensor
        )
    """
    feature_map = {
        'filename': tf.FixedLenFeature([], dtype=tf.string),
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], dtype=tf.int64)
    }
    features = tf.parse_single_example(raw_record, feature_map)
    image_bytes = tf.decode_raw(features["image"], tf.uint8)
    filename = tf.cast(features['filename'], tf.string)
    label = tf.strings.substr(filename, -10, 1)
    label = tf.strings.regex_replace(label, 'b', '0')
    label = tf.strings.regex_replace(label, 'm', '1')
    label = tf.strings.to_number(label, out_type=tf.int32)
    label = tf.one_hot(label, 2)

    orig_image = tf.reshape(
        image_bytes, (512, 512, 3)
    )

    image = _preprocess_image(
        image=orig_image,
        is_training=is_training,
        new_size=new_size,
    )

    return image, label, orig_image


def _parse_record_z(raw_record, is_training, new_size=64):
    """Parse a single instance record with differential recognition array

    Parameters:
        raw_record (tf.data.example): a raw record from a tf.record
        is_training (bool): Whether to perfrom data augmentations on record
        new_size (int): Size to crop image to

    Returns:
    tuple(
        image (tf.tensor): a tf.tensor w/ shape [new_size,new_size,3]
        label (tf.tensor): a one-hot encoded tf.tensor
        z_array (tf.tensor): a 128 length tf.float32 tensor of the
        differential z_score layer
        )
    """

    feature_map = {
        'img': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], dtype=tf.int64),
        'z_array': tf.FixedLenFeature([128], dtype=tf.float32)
    }
    features = tf.parse_single_example(raw_record, feature_map)
    image_bytes = tf.decode_raw(features["img"], tf.uint8)
    z_array = features['z_array']
    label = features['label']
    label = tf.one_hot(label, 2)
    image = tf.reshape(
        image_bytes, (512, 512, 3)
    )

    image = _preprocess_image(
        image=image,
        is_training=is_training,
        new_size=new_size,
    )

    return image, label, z_array


def create_dataset(filenames, batch_size, augment, z_layer,
                   drop_remainder=False):
    """Creates a tf.data.Dataset from a list of .record filenames

    Parameters:
        filenames (string[]): A list of .record filenames
        z_array (bool): (Optional.) bool representing whether the tfrecords
            include z-score layers.
        augment (bool): bool representing whether to perform random data
            augmentations
        batch_size (int): Batch size of dataset
        drop_remainder (bool): (Optional.) bool representing whether the
            last batch should be dropped in the case it has fewer than
            `batch_size` elements; the default behavior is not to drop the
            smaller batch.
    Returns:
        dataset: A 'tf.data.Dataset'
    """
    filenames = tf.data.Dataset.list_files(filenames, shuffle=True)
    raw_dataset = tf.data.TFRecordDataset(filenames)
    if z_layer:
        parse_fun = _parse_record_z
    else:
        parse_fun = _parse_record_orig

    dataset = raw_dataset.map(lambda x: parse_fun(x, augment),
                              num_parallel_calls=10)
    dataset = \
        dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset
