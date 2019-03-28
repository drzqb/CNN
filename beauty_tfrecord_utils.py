'''
    beauty image data to tfrecord file
'''
import tensorflow as tf
import os
from PIL import Image
import numpy as np
from tensorflow.contrib.data import AUTOTUNE

IMAGE_SIZE = 96
IMAGE_CHANNEL = 3


def write2tfrecord(data_path, suffix):
    tfrecord_name = 'data/' + suffix + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    files = os.listdir(data_path)
    r = np.random.permutation(len(files))
    files = [files[i] for i in r]
    for image_filename in files:
        if image_filename.endswith('.jpg'):
            arr = Image.open(os.path.join(data_path, image_filename))
            arr_raw = arr.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw]))
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


def single_example_parser(serialized_example):
    features_parsed = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string)
        }
    )

    image = features_parsed['image']

    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    image = tf.cast(image, tf.float32) / 127.5 - 1

    return image


def batched_data(tfrecord_filename, single_example_parser, batch_size, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .batch(batch_size, drop_remainder=True) \
        .repeat() \
        .prefetch(buffer_size=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    write2tfrecord('data/faces/', 'beauty')

    sess = tf.Session()
    data_batch = batched_data(['data/beauty.tfrecord'], single_example_parser, 10)
    data_batch_ = sess.run(data_batch)

    print(data_batch_)
