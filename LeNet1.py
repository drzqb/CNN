'''
    LeNet for mnist
'''
import tensorflow as tf
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 100, 'number of iterations')
tf.flags.DEFINE_string('model_save_path', 'model/lenet1', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.1, 'learning rate for training')
tf.flags.DEFINE_integer('per_save', 10, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS

WIDTH, HEIGHT = 28, 28
CHANNEL = 1
CLASSIFY = 10


class LeNet():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            self.image = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNEL], name='image')
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')

        with tf.name_scope('conv'):
            conv1 = tf.layers.Conv2D(32, 5, padding='same', activation='relu')(self.image)
            pool1 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv1)

            conv2 = tf.layers.Conv2D(64, 5, padding='same', activation='relu')(pool1)
            pool2 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv2)

        with tf.name_scope('flatten'):
            flatten = tf.layers.Flatten()(pool2)

        with tf.name_scope('output'):
            fc = tf.layers.Dense(512, activation='relu')(flatten)
            self.output = tf.layers.Dense(CLASSIFY)(fc)

        with tf.name_scope('loss'):
            self.prediction = tf.argmax(self.output, axis=-1, output_type=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.label), tf.float32))

            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.output))

        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sess, X, Y):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.train_op], feed_dict={
            self.image: X,
            self.label: Y,
        })
        return loss, accuracy

    def eval(self, sess, X, Y):
        accuracy = sess.run(self.accuracy, feed_dict={
            self.image: X,
            self.label: Y,
        })
        return accuracy

    def predict(self, sess, X):
        result = sess.run(self.prediction, feed_dict={
            self.image: X
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, save_path=path)


class Data():
    @staticmethod
    def load_data():
        mnist = input_data.read_data_sets('data/MNIST_data/')
        return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    @staticmethod
    def get_batch(X, Y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        return X[start:end], Y[start:end]

    @staticmethod
    def shuffle(X, Y):
        r = np.random.permutation(np.shape(X)[0])
        return X[r], Y[r]


class User():
    @staticmethod
    def train(X):
        trX, trY, teX, teY = X[0], X[1], X[2], X[3]
        with tf.Session() as sess:
            mylenet = LeNet(CONFIG)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            m_samples = np.shape(trX)[0]
            total_batch = m_samples // CONFIG.batch_size

            if CONFIG.mode == 'train1':
                mylenet.restore(sess, CONFIG.model_save_path)

            loss = []
            acc = []
            for epoch in range(1, CONFIG.epochs + 1):
                loss_epoch = 0.0
                acc_epoch = 0.0

                for batch in range(total_batch):
                    X_batch, Y_batch = Data.get_batch(trX, trY, CONFIG.batch_size, batch)

                    loss_batch, acc_batch = mylenet.train(sess, X_batch, Y_batch)
                    loss_epoch += loss_batch
                    acc_epoch += acc_batch

                    sys.stdout.write('\r>> %d/%d | %d/%d | loss_batch: %f  acc_batch:%.2f%%' % (
                        epoch, CONFIG.epochs, batch + 1, total_batch, loss_batch, 100.0 * acc_batch))
                    sys.stdout.flush()
                loss.append(loss_epoch / total_batch)
                acc.append(acc_epoch / total_batch)

                sys.stdout.write(' | loss: %f  acc:%.2f%%' % (loss[-1], 100.0 * acc[-1]))
                sys.stdout.flush()

                acc_val = mylenet.eval(sess, teX, teY)
                sys.stdout.write(' | acc_val:%.2f%%\n' % (100.0 * acc_val))
                sys.stdout.flush()

                teX, teY = Data.shuffle(teX, teY)

                if epoch % CONFIG.per_save == 0:
                    mylenet.save(sess, CONFIG.model_save_path)

    @staticmethod
    def predict(X):
        with tf.Session() as sess:
            mylenet = LeNet(CONFIG)
            mylenet.restore(sess, CONFIG.model_save_path)

            result = mylenet.predict(sess, X)
            print(result)
            print('----------------------------------------------------------------')


def main(unused_argvs):
    if CONFIG.mode.startswith('train'):
        trX, trY, teX, teY = Data.load_data()
        trX = np.reshape(trX, [-1, WIDTH, HEIGHT, CHANNEL])
        teX = np.reshape(teX, [-1, WIDTH, HEIGHT, CHANNEL])
        User.train([trX, trY, teX, teY])
    elif CONFIG.mode == 'predict':
        _, _, teX, _ = Data.load_data()
        teX = np.reshape(teX, [-1, WIDTH, HEIGHT, CHANNEL])
        User.predict(teX)


if __name__ == '__main__':
    tf.app.run()
