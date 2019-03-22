'''
    AlexNet for catVSdog without using two GPUs
'''
import tensorflow as tf
from cad_tfrecord_utils import batched_data, single_example_parser

tf.flags.DEFINE_integer('batch_size', 32, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 100000, 'number of iterations')
tf.flags.DEFINE_string('model_save_path', 'model/alexnet1', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep probility for rnn outputs')

tf.flags.DEFINE_integer('per_save', 100, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train1', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS

WIDTH, HEIGHT = 224, 224
CHANNEL = 3
CLASSIFY = 2


class AlexNet():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            self.image = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNEL], name='image')
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('conv'):
            conv1 = tf.layers.Conv2D(64, 11, strides=[4, 4], padding='same', activation='relu',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                     )(self.image)
            pool1 = tf.layers.MaxPooling2D(3, 2, padding='valid')(conv1)
            lrn1 = tf.nn.lrn(pool1, 4, 1, 1e-3 / 9.0, 0.75)

            conv2 = tf.layers.Conv2D(192, 5, padding='same', activation='relu',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                     )(lrn1)
            pool2 = tf.layers.MaxPooling2D(3, 2, padding='valid')(conv2)
            lrn2 = tf.nn.lrn(pool2, 4, 1, 1e-3 / 9.0, 0.75)

            conv3 = tf.layers.Conv2D(384, 3, padding='same', activation='relu',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                     )(lrn2)

            conv4 = tf.layers.Conv2D(256, 3, padding='same', activation='relu',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                     )(conv3)

            conv5 = tf.layers.Conv2D(256, 3, padding='same', activation='relu',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                     )(conv4)
            pool5 = tf.layers.MaxPooling2D(3, 2, padding='valid')(conv5)

        with tf.name_scope('flatten'):
            flatten = tf.layers.Flatten()(pool5)

        with tf.name_scope('fc'):
            fc1 = tf.nn.dropout(tf.layers.Dense(4096, activation='relu',
                                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                )(flatten),
                                keep_prob=self.keep_prob)
            fc2 = tf.nn.dropout(tf.layers.Dense(4096, activation='relu',
                                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                )(fc1),
                                keep_prob=self.keep_prob)

        with tf.name_scope('output'):
            self.output = tf.layers.Dense(CLASSIFY,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer()
                                          )(fc2)

        with tf.name_scope('loss'):
            self.prediction = tf.argmax(self.output, axis=-1, output_type=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.label), tf.float32))

            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.output))

        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sess, X, Y):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.train_op], feed_dict={
            self.image: X,
            self.label: Y,
            self.keep_prob: self.config.keep_prob
        })
        return loss, accuracy

    def eval(self, sess, X, Y):
        accuracy = sess.run(self.accuracy, feed_dict={
            self.image: X,
            self.label: Y,
            self.keep_prob: 1.0
        })
        return accuracy

    def predict(self, sess, X):
        result = sess.run(self.prediction, feed_dict={
            self.image: X,
            self.keep_prob: 1.0
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, save_path=path)


class User():
    @staticmethod
    def train():
        train_file = ['data/train_cad.tfrecord']
        valid_file = ['data/val_cad.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, CONFIG.batch_size, 10 * CONFIG.batch_size)
        valid_batch = batched_data(valid_file, single_example_parser, 100, shuffle=False)

        with tf.Session() as sess:
            myalexnet = AlexNet(CONFIG)

            if CONFIG.mode == 'train0':
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            elif CONFIG.mode == 'train1':
                myalexnet.restore(sess, CONFIG.model_save_path)

            X_val = sess.run(valid_batch)
            loss = []
            acc = []
            for epoch in range(1, CONFIG.epochs + 1):
                X = sess.run(train_batch)

                loss_, acc_ = myalexnet.train(sess, X[0], X[1])
                loss.append(loss_)
                acc.append(acc_)

                print('>> %d/%d | loss: %f  acc: %.2f%%' % (epoch, CONFIG.epochs, loss_, 100.0 * acc_))

                if epoch % CONFIG.per_save == 0:
                    acc_val = myalexnet.eval(sess, X_val[0], X_val[1])
                    print(' acc_val: %.2f%%\n' % (100.0 * acc_val))

                    myalexnet.save(sess, CONFIG.model_save_path)

    @staticmethod
    def predict():
        valid_file = ['data/val_cad.tfrecord']
        valid_batch = batched_data(valid_file, single_example_parser, 100, shuffle=False)

        with tf.Session() as sess:
            myalexnet = AlexNet(CONFIG)
            myalexnet.restore(sess, CONFIG.model_save_path)

            X_val = sess.run(valid_batch)

            result = myalexnet.predict(sess, X_val[0])
            print(result)
            print('----------------------------------------------------------------')


def main(unused_argvs):
    if CONFIG.mode.startswith('train'):
        User.train()
    elif CONFIG.mode == 'predict':
        User.predict()


if __name__ == '__main__':
    tf.app.run()
