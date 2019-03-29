'''
    CGAN for mnist
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
from mnist_tfrecord_utils import batched_data, single_example_parser

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('noise_dim', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 1000000, 'number of iterations')
tf.flags.DEFINE_string('model_save_path', 'model/gan6/', 'directory of model file saved')
tf.flags.DEFINE_float('lr_d', 0.0001, 'learning rate for discriminator')
tf.flags.DEFINE_float('lr_g', 0.0001, 'learning rate for generator')
tf.flags.DEFINE_float('beta_d', 0.5, 'beta1 for adamoptimizer of discriminator')
tf.flags.DEFINE_float('beta_g', 0.5, 'beta1 for adamoptimizer of generator')
tf.flags.DEFINE_integer('per_save', 1000, 'save model once every per_save iterations')
tf.flags.DEFINE_string('loss_mode', 'Log', 'mode of loss:'
                                           'LS'
                                           'Log')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS

IMAGE_SIZE = 28
IMAGE_SIZE2 = IMAGE_SIZE ** 2
N_CLASS = 10
STDDEV = 0.02


class CGAN(object):
    def __init__(self, config):
        self.config = config

        self.build_model()

    def generator(self, noise, c, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            out = tf.layers.dense(tf.concat([noise, c], axis=-1), 64,
                                  activation='relu',
                                  kernel_initializer=tf.random_normal_initializer(stddev=STDDEV))
            out = tf.layers.dense(tf.concat([out, c], axis=-1), IMAGE_SIZE2,
                                  activation='sigmoid',
                                  kernel_initializer=tf.random_normal_initializer(stddev=STDDEV))

            return out

    def discriminator(self, x, c, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            out = tf.layers.dense(tf.concat([x, c], axis=-1), 64,
                                  activation='relu',
                                  kernel_initializer=tf.random_normal_initializer(stddev=STDDEV))
            out = tf.layers.dense(tf.concat([out, c], axis=-1), 1,
                                  activation='sigmoid',
                                  kernel_initializer=tf.random_normal_initializer(stddev=STDDEV))
            return out

    def build_model(self):
        self.X = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE2], name='X')
        self.C = tf.placeholder(tf.int32, shape=[None], name='C')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.config.noise_dim], name='Z')

        C_onehot = tf.one_hot(self.C, N_CLASS)
        self.fake_image = self.generator(self.Z, C_onehot)
        D_fake = self.discriminator(self.fake_image, C_onehot)
        D_real = self.discriminator(self.X, C_onehot, reuse=True)

        # loss includes 'LS' and 'Log'
        if self.config.loss_mode == 'LS':
            self.D_loss = tf.reduce_mean((D_real - 1.0) ** 2 + D_fake ** 2)
            self.G_loss = tf.reduce_mean((D_fake - 1.0) ** 2)
        else:
            self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
            self.G_loss = -tf.reduce_mean(tf.log(D_fake))

        var_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        var_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.D_train_op = tf.train.AdamOptimizer(self.config.lr_d, beta1=self.config.beta_d).minimize(self.D_loss,
                                                                                                      var_list=var_D)
        self.G_train_op = tf.train.AdamOptimizer(self.config.lr_g, beta1=self.config.beta_g).minimize(self.G_loss,
                                                                                                      var_list=var_G)

    def train_D(self, sess, X, C, Z):
        _, D_loss_curr = sess.run([self.D_train_op, self.D_loss], feed_dict={self.X: X, self.C: C, self.Z: Z})

        return D_loss_curr

    def train_G(self, sess, C, Z):
        _, G_loss_curr = sess.run([self.G_train_op, self.G_loss], feed_dict={self.C: C, self.Z: Z})

        return G_loss_curr

    def predict(self, sess, C, Z):
        result = sess.run(self.fake_image, feed_dict={self.C: C, self.Z: Z})
        return result

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, save_path=path)


class Data():
    @staticmethod
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size=[m, n])


class Plot():
    @staticmethod
    def plot_image(samples, suffix=None):
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(5, 10)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(IMAGE_SIZE, IMAGE_SIZE))

        if suffix is not None:
            plt.savefig(CONFIG.model_save_path + suffix + '.png', bbox_inches='tight')
        else:
            plt.savefig(CONFIG.model_save_path + 'predict.png', bbox_inches='tight')

        plt.close(fig)

    @staticmethod
    def plot_loss(g_loss, d_loss):
        plt.plot(g_loss, label='G_Loss')
        plt.plot(d_loss, label='D_Loss')
        plt.legend(loc='upper right')
        plt.savefig(CONFIG.model_save_path + 'Loss.png', bbox_inches='tight')
        plt.close()


class User():
    @staticmethod
    def train():
        train_file = ['data/mnist.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, CONFIG.batch_size, 10 * CONFIG.batch_size)

        with tf.Session() as sess:
            mycgan = CGAN(CONFIG)
            if CONFIG.mode == 'train0':
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            elif CONFIG.mode == 'train1':
                mycgan.restore(sess, CONFIG.model_save_path)

            number_trainable_variables = 0
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            for k, v in zip(variable_names, values):
                print("Variable: ", k)
                print("Shape: ", v.shape)
                number_trainable_variables += np.prod([s for s in v.shape])

            print('Total number of parameters: %d' % number_trainable_variables)
            d_loss = []
            g_loss = []

            print('训练开始时间：  %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            for it in range(0, CONFIG.epochs):
                if it % CONFIG.per_save == 0 or it == CONFIG.epochs - 1:
                    samples = mycgan.predict(sess,
                                             np.tile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5]),
                                             Data.sample_Z(50, CONFIG.noise_dim))
                    Plot.plot_image(samples, str(it))

                    mycgan.save(sess, CONFIG.model_save_path)

                X_mb, C_mb = sess.run(train_batch)
                Z_mb = Data.sample_Z(CONFIG.batch_size, CONFIG.noise_dim)
                D_loss_curr = mycgan.train_D(sess, X_mb, C_mb, Z_mb)
                G_loss_curr = mycgan.train_G(sess, C_mb, Z_mb)
                G_loss_curr = mycgan.train_G(sess, C_mb, Z_mb)

                d_loss.append(D_loss_curr)
                g_loss.append(G_loss_curr)

                print('%d/%d' % (it + 1, CONFIG.epochs), '|    D_loss:', d_loss[-1], '    G_loss:', g_loss[-1],
                      '  %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            print('训练结束时间：  %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            Plot.plot_loss(g_loss, d_loss)

    @staticmethod
    def predict():
        with tf.Session() as sess:
            mycgan = CGAN(CONFIG)
            mycgan.restore(sess, CONFIG.model_save_path)

            samples = mycgan.predict(sess,
                                     np.tile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5]),
                                     Data.sample_Z(50, CONFIG.noise_dim))
            Plot.plot_image(samples)


def main(unused_argvs):
    if not os.path.exists(CONFIG.model_save_path):
        os.makedirs(CONFIG.model_save_path)

    if CONFIG.mode.startswith('train'):
        User.train()
    elif CONFIG.mode == 'predict':
        User.predict()


if __name__ == '__main__':
    tf.app.run()
