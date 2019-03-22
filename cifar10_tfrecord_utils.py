'''
    cifar10 image data to tfrecord file
'''
import tensorflow as tf
import os
import numpy as np
import pickle

IMAGE_SIZE = 32
IMAGE_CHANNEL = 3


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    Xtr = np.zeros((50000, 3, 32, 32))
    Ytr = np.zeros(50000, dtype=np.int)
    for b in range(1, 6):
        Xtr[(b - 1) * 10000:b * 10000], Ytr[(b - 1) * 10000:b * 10000] = load_CIFAR_batch(
            os.path.join(ROOT, 'data_batch_%d' % (b)))
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return np.transpose(Xtr, [0, 2, 3, 1]), Ytr, np.transpose(Xte, [0, 2, 3, 1]), Yte


def write2tfrecord(image, label, data_usage, suffix):
    tfrecord_name = 'data/' + data_usage + '_' + suffix + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    m_samples = np.shape(image)[0]
    for i in range(m_samples):
        arr_raw = image[i].tobytes()
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr_raw]))
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
    image = tf.decode_raw(image, tf.float64)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5

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
    trX, trY, teX, teY = load_CIFAR10('data\cifar-10-batches-py')

    write2tfrecord(trX, trY, 'train', 'cifar10')
    write2tfrecord(teX, teY, 'val', 'cifar10')

    sess = tf.Session()
    data_batch = batched_data(['data/train_cifar10.tfrecord'], single_example_parser, 10)
    data_batch_ = sess.run(data_batch)

    print(data_batch_[0])
    print(data_batch_[1])
