'''
    catVSdog image data to tfrecord file
'''
import tensorflow as tf
import os
from PIL import Image
import numpy as np

IMAGE_SIZE = 224
IMAGE_CHANNEL = 3


def write2tfrecord(data_path, data_usage, suffix):
    tfrecord_name = 'data/' + data_usage + '_' + suffix + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    files = os.listdir(data_path)
    r = np.random.permutation(len(files))
    files = [files[i] for i in r]
    for image_filename in files:
        if image_filename.endswith('.jpg'):
            if image_filename.startswith('cat'):
                index = 0
            else:
                index = 1
            arr = Image.open(os.path.join(data_path, image_filename))
            arr = arr.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            arr_raw = arr.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw])),
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串

    writer.close()


def single_example_parser(serialized_example):
    features_parsed = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        }
    )

    label = features_parsed['label']
    image = features_parsed['image']

    label = tf.cast(label, tf.int32)
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def batched_data(tfrecord_filename, single_example_parser, batch_size, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename) \
        .map(single_example_parser) \
        .batch(batch_size) \
        .repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    # write2tfrecord('data/cad/train/', 'train', 'cad')
    # write2tfrecord('data/cad/val/', 'val', 'cad')

    sess = tf.Session()
    data_batch = batched_data(['data/train_cad.tfrecord'], single_example_parser, 10)
    data_batch_ = sess.run(data_batch)

    print(data_batch_[0])
    print(data_batch_[1])
