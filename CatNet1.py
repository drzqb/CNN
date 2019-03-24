'''
    CatNet based on VGG16 for cifar10
'''
import tensorflow as tf
from cifar10_tfrecord_utils import batched_data, single_example_parser

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 100000, 'number of iterations')
tf.flags.DEFINE_string('model_save_path', 'model/CatNet1', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')

tf.flags.DEFINE_integer('per_save', 100, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS

WIDTH, HEIGHT = 32, 32
CHANNEL = 3
CLASSIFY = 10


class CatNet():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        with tf.name_scope('input'):
            self.image = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNEL], name='image')
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')

        with tf.name_scope('conv1'):
            conv1_1 = tf.layers.Conv2D(16, 3, padding='same', activation='relu',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                       )(self.image)
            conv1_2 = tf.layers.Conv2D(16, 3, padding='same', activation='relu',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                       )(conv1_1)

        with tf.name_scope('conv2'):
            conv2_1 = tf.layers.Conv2D(32, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv1_2)
            conv2_2 = tf.layers.Conv2D(32, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv2_1)
            pool2 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv2_2)

        with tf.name_scope('conv3'):
            conv3_1 = tf.layers.Conv2D(64, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(pool2)
            conv3_2 = tf.layers.Conv2D(32, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv3_1)

        with tf.name_scope('conv4'):
            conv4_1 = tf.layers.Conv2D(128, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv3_2)
            conv4_2 = tf.layers.Conv2D(128, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv4_1)
            conv4_3 = tf.layers.Conv2D(128, 3, padding='same', activation='relu',
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                                     )(conv4_2)
            pool4 = tf.layers.MaxPooling2D(2, 2, padding='same')(conv4_3)

        with tf.name_scope('flatten'):
            flatten1 = tf.layers.Flatten()(conv1_2)
            flatten2 = tf.layers.Flatten()(pool2)
            flatten3 = tf.layers.Flatten()(conv3_2)
            flatten4 = tf.layers.Flatten()(pool4)
            flatten=tf.concat([flatten1,flatten2,flatten3,flatten4],axis=-1)

        with tf.name_scope('output'):
            self.output = tf.layers.Dense(CLASSIFY,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer()
                                          )(flatten)

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
        })
        return loss, accuracy, prediction

    def eval(self, sess, X, Y):
        accuracy = sess.run(self.accuracy, feed_dict={
            self.image: X,
            self.label: Y,
        })
        return accuracy

    def predict(self, sess, X):
        result = sess.run(self.prediction, feed_dict={
            self.image: X,
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
        train_file = ['data/train_cifar10.tfrecord']
        valid_file = ['data/val_cifar10.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, CONFIG.batch_size, 10 * CONFIG.batch_size)
        valid_batch = batched_data(valid_file, single_example_parser, 100, shuffle=False)

        with tf.Session() as sess:
            mycatnet = CatNet(CONFIG)

            if CONFIG.mode == 'train0':
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            elif CONFIG.mode == 'train1':
                mycatnet.restore(sess, CONFIG.model_save_path)

            X_val = sess.run(valid_batch)

            loss = []
            acc = []
            for epoch in range(1, CONFIG.epochs + 1):
                X = sess.run(train_batch)

                loss_, acc_, prediction_ = mycatnet.train(sess, X[0], X[1])
                loss.append(loss_)
                acc.append(acc_)

                print('>> %d/%d | loss: %f  acc: %.2f%%' % (epoch, CONFIG.epochs, loss_, 100.0 * acc_))
                print(prediction_)
                print(X[1])
                if epoch % CONFIG.per_save == 0:
                    acc_val = mycatnet.eval(sess, X_val[0], X_val[1])
                    print(' acc_val: %.2f%%\n' % (100.0 * acc_val))

                    mycatnet.save(sess, CONFIG.model_save_path)

    @staticmethod
    def predict():
        valid_file = ['data/val_cifar10.tfrecord']
        valid_batch = batched_data(valid_file, single_example_parser, 100, shuffle=False)

        with tf.Session() as sess:
            mycatnet = CatNet(CONFIG)
            mycatnet.restore(sess, CONFIG.model_save_path)

            X_val = sess.run(valid_batch)

            result = mycatnet.predict(sess, X_val[0])
            print(result)
            print('----------------------------------------------------------------')


def main(unused_argvs):
    if CONFIG.mode.startswith('train'):
        User.train()
    elif CONFIG.mode == 'predict':
        User.predict()


if __name__ == '__main__':
    tf.app.run()
