'''
    VGG16 for catVSdog
'''
import tensorflow as tf
from cad_tfrecord_utils import batched_data, single_example_parser

tf.flags.DEFINE_integer('batch_size', 32, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 100000, 'number of iterations')
tf.flags.DEFINE_string('model_save_path', 'model/VGG161', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep probility for rnn outputs')

tf.flags.DEFINE_integer('per_save', 100, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS

WIDTH, HEIGHT = 224, 224
CHANNEL = 3
CLASSIFY = 2


class VGG16():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            self.image = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNEL], name='image')
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope('conv1'):
            conv1_1 = tf.layers.Conv2D(16, 3, padding='same', activation='relu',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                       )(self.image)
            conv1_2 = tf.layers.Conv2D(16, 3, padding='same', activation='relu',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                       )(conv1_1)
            pool1 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv1_2)

        with tf.name_scope('conv2'):
            conv2_1 = tf.nn.dropout(tf.layers.Conv2D(32, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(pool1), self.keep_prob)
            conv2_2 = tf.nn.dropout(tf.layers.Conv2D(32, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv2_1), self.keep_prob)
            pool2 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv2_2)

        with tf.name_scope('conv3'):
            conv3_1 = tf.nn.dropout(tf.layers.Conv2D(64, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(pool2), self.keep_prob)
            conv3_2 = tf.nn.dropout(tf.layers.Conv2D(32, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv3_1), self.keep_prob)
            pool3 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv3_2)

        with tf.name_scope('conv4'):
            conv4_1 = tf.nn.dropout(tf.layers.Conv2D(128, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(pool3), self.keep_prob)
            conv4_2 = tf.nn.dropout(tf.layers.Conv2D(128, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv4_1), self.keep_prob)
            conv4_3 = tf.nn.dropout(tf.layers.Conv2D(128, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv4_2), self.keep_prob)
            pool4 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv4_3)

        with tf.name_scope('flatten'):
            flatten = tf.layers.Flatten()(pool4)

        with tf.name_scope('fc'):
            fc1 = tf.nn.dropout(tf.layers.Dense(1024, activation='relu',
                                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                )(flatten),
                                keep_prob=self.keep_prob)
            fc2 = tf.nn.dropout(tf.layers.Dense(1024, activation='relu',
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

        optimizer = tf.train.MomentumOptimizer(self.config.lr, 0.9)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sess, X, Y):
        loss, accuracy, prediction, _ = sess.run([self.loss, self.accuracy, self.prediction, self.train_op], feed_dict={
            self.image: X,
            self.label: Y,
            self.keep_prob: self.config.keep_prob
        })
        return loss, accuracy, prediction

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
            myvgg16 = VGG16(CONFIG)

            if CONFIG.mode == 'train0':
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            elif CONFIG.mode == 'train1':
                myvgg16.restore(sess, CONFIG.model_save_path)

            X_val = sess.run(valid_batch)
            loss = []
            acc = []
            for epoch in range(1, CONFIG.epochs + 1):
                X = sess.run(train_batch)

                loss_, acc_, prediction_ = myvgg16.train(sess, X[0], X[1])
                loss.append(loss_)
                acc.append(acc_)

                print('>> %d/%d | loss: %f  acc: %.2f%%' % (epoch, CONFIG.epochs, loss_, 100.0 * acc_))
                print(prediction_)
                print(X[1])
                if epoch % CONFIG.per_save == 0:
                    acc_val = myvgg16.eval(sess, X_val[0], X_val[1])
                    print(' acc_val: %.2f%%\n' % (100.0 * acc_val))

                    myvgg16.save(sess, CONFIG.model_save_path)

    @staticmethod
    def predict():
        valid_file = ['data/val_cad.tfrecord']
        valid_batch = batched_data(valid_file, single_example_parser, 100, shuffle=False)

        with tf.Session() as sess:
            myvgg16 = VGG16(CONFIG)
            myvgg16.restore(sess, CONFIG.model_save_path)

            X_val = sess.run(valid_batch)

            result = myvgg16.predict(sess, X_val[0])
            print(result)
            print('----------------------------------------------------------------')


def main(unused_argvs):
    if CONFIG.mode.startswith('train'):
        User.train()
    elif CONFIG.mode == 'predict':
        User.predict()


if __name__ == '__main__':
    tf.app.run()
