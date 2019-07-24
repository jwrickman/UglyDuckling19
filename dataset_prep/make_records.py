import tensorflow as tf
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(img, label, z):
    img_raw = img.tostring()
    label = int(label[1])
    feature_dict = {
        'img': _bytes_feature(img_raw),
        'label': _int64_feature(label),
        'z_array': _float_feature(z)}
    example =\
        tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(img, lable, z, out_path, filename, overwrite=False):
    path = out_path + "/" + filename
    if not os.path.exists(path) or overwrite:
        writer = \
            tf.python_io.TFRecordWriter(path)
        tf_example = create_tf_example(img, lable, z)
        writer.write(tf_example.SerializeToString())
        writer.close()
        return tf_example
