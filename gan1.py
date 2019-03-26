'''
    DCGAN for beauty
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from beauty_tfrecord_utils import batched_data, single_example_parser

tf.flags.DEFINE_integer('batch_size', 32, 'batch size for training')
tf.flags.DEFINE_integer('noise_dim', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 1000000, 'number of iterations')
tf.flags.DEFINE_string('model_save_path', 'model/', 'directory of model file saved')
tf.flags.DEFINE_float('lr_d', 0.0001, 'learning rate for discriminator')
tf.flags.DEFINE_float('lr_g', 0.0001, 'learning rate for generator')
tf.flags.DEFINE_float('beta_d', 0.5, 'beta1 for adamoptimizer of discriminator')
tf.flags.DEFINE_float('beta_g', 0.5, 'beta1 for adamoptimizer of generator')
tf.flags.DEFINE_integer('per_save', 1000, 'save model once every per_save iterations')
tf.flags.DEFINE_string('loss_mode', 'LS', 'mode of loss:'
                                          'LS'
                                          'Log')
tf.flags.DEFINE_string('mode', 'train0', 'The mode of train or predict as follows: '
                                         'train0: train first time or retrain'
                                         'train1: continue train'
                                         'predict: predict')

CONFIG = tf.flags.FLAGS

IMAGE_SIZE = 96
IMAGE_CHANNEL = 3
EPS = 1.0e-10


class DCGAN(object):
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        self.variables_discriminater()
        self.variables_generator()
        self.Loss_Optim()

    def train_D(self, sess, X, Z):
        _, D_loss_curr = sess.run([self.D_train_op, self.D_loss], feed_dict={self.X: X, self.Z: Z})

        return D_loss_curr

    def train_G(self, sess, Z):
        _, G_loss_curr = sess.run([self.G_train_op, self.G_loss], feed_dict={self.Z: Z})

        return G_loss_curr

    def predict(self, sess, Z):
        result = sess.run(self.fake_image, feed_dict={self.Z: Z})
        return result

    def save(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver(max_to_keep=1)
        saver.restore(sess, save_path=path)

    def weight_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_var(self, shape, name, value=0.0):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value))

    def variables_discriminater(self):
        self.X = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL],
                                name='X')

        self.W_d1 = self.weight_var([4, 4, IMAGE_CHANNEL, 32], 'W_d1')
        self.b_d1 = self.bias_var([32], 'b_d1')

        self.W_d2 = self.weight_var([(IMAGE_SIZE // 2) * (IMAGE_SIZE // 2) * 32, 100], 'W_d2')
        self.b_d2 = self.bias_var([100], 'b_d2')
        self.beta_d2 = self.bias_var([100], 'beta_d2')
        self.gamma_d2 = self.bias_var([100], 'gamma_d2', 1.0)

        self.W_d3 = self.weight_var([100, 1], 'W_d3')
        self.b_d3 = self.bias_var([1], 'b_d3')

        self.theta_D = [self.W_d1, self.b_d1, self.W_d2, self.b_d2, self.beta_d2, self.gamma_d2,
                        self.W_d3, self.b_d3]

    def discriminator(self, x):
        out_1 = tf.nn.conv2d(x, self.W_d1, strides=[1, 2, 2, 1], padding='SAME')
        out_1 = tf.nn.bias_add(out_1, self.b_d1)
        out_1 = tf.reshape(out_1, [-1, (IMAGE_SIZE // 2) * (IMAGE_SIZE // 2) * 32])
        out_2 = tf.maximum(0.2 * out_1, out_1)

        out_3 = tf.matmul(out_2, self.W_d2) + self.b_d2
        batch_mean, batch_var = tf.nn.moments(out_3, [0])
        out_3 = tf.nn.batch_normalization(out_3, batch_mean, batch_var, self.beta_d2, self.gamma_d2, EPS)
        out_4 = tf.maximum(0.2 * out_3, out_3)

        out_5 = tf.matmul(out_4, self.W_d3) + self.b_d3
        out_5 = tf.sigmoid(out_5)

        return out_5

    def variables_generator(self):
        self.Z = tf.placeholder(tf.float32, shape=[None, self.config.noise_dim], name='Z')

        self.W_g1 = self.weight_var([self.config.noise_dim, 8 * IMAGE_SIZE * IMAGE_SIZE], 'W_g1')
        self.b_g1 = self.bias_var([8 * IMAGE_SIZE * IMAGE_SIZE], 'b_g1')
        self.beta_g1 = self.bias_var([32], 'beta_g1')
        self.gamma_g1 = self.bias_var([32], 'gamma_g1', 1.0)

        self.W_g2 = self.weight_var([5, 5, IMAGE_CHANNEL, 32], 'W_g2')
        self.b_g2 = self.bias_var([IMAGE_CHANNEL], 'b_g2')

        self.theta_G = [self.W_g1, self.b_g1, self.beta_g1, self.gamma_g1,
                        self.W_g2, self.b_g2]

    def generator(self, noise):
        out_1 = tf.matmul(noise, self.W_g1) + self.b_g1
        out_1 = tf.reshape(out_1, [-1, IMAGE_SIZE // 2, IMAGE_SIZE // 2, 32])
        batch_mean, batch_var = tf.nn.moments(out_1, [0, 1, 2])
        out_1 = tf.nn.batch_normalization(out_1, batch_mean, batch_var, self.beta_g1, self.gamma_g1, EPS)
        out_1 = tf.nn.relu(out_1)

        out_2 = tf.nn.conv2d_transpose(out_1, self.W_g2, output_shape=tf.stack(
            [tf.shape(out_1)[0], IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]), strides=[1, 2, 2, 1],
                                       padding='SAME')
        out_2 = tf.nn.bias_add(out_2, self.b_g2)
        out_2 = tf.nn.tanh(out_2)

        return out_2

    def Loss_Optim(self):
        self.fake_image = self.generator(self.Z)
        D_fake = self.discriminator(self.fake_image)
        D_real = self.discriminator(self.X)

        # loss includes 'LS' and 'Log'
        if self.config.loss_mode is None or self.config.loss_mode == 'LS':
            self.D_loss = tf.reduce_mean((D_real - 1.0) ** 2 + D_fake ** 2)
            self.G_loss = tf.reduce_mean((D_fake - 1.0) ** 2)
        else:
            self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
            self.G_loss = -tf.reduce_mean(tf.log(D_fake))

        self.D_train_op = tf.train.AdamOptimizer(self.config.lr_d, beta1=self.config.beta_d).minimize(self.D_loss,
                                                                                                      var_list=self.theta_D)
        self.G_train_op = tf.train.AdamOptimizer(self.config.lr_g, beta1=self.config.beta_g).minimize(self.G_loss,
                                                                                                      var_list=self.theta_G)


class Data():
    @staticmethod
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size=[m, n])


class Plot():
    @staticmethod
    def plot_image(self, samples, suffix=None):
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample)

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
        train_file = ['data/beauty.tfrecord']

        train_batch = batched_data(train_file, single_example_parser, CONFIG.batch_size, 10 * CONFIG.batch_size)

        with tf.Session() as sess:
            mydcgan = DCGAN(CONFIG)
            if CONFIG.mode == 'train0':
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            elif CONFIG.mode == 'train1':
                mydcgan.restore(sess, CONFIG.model_save_path)

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
            for it in range(1, CONFIG.epochs + 1):
                X_mb = sess.run(train_batch)
                Z_mb = Data.sample_Z(CONFIG.batch_size, CONFIG.noise_dim)
                D_loss_curr = mydcgan.train_D(sess, X_mb, Z_mb)
                G_loss_curr = mydcgan.train_G(sess, Z_mb)
                G_loss_curr = mydcgan.train_G(sess, Z_mb)

                d_loss.append(D_loss_curr)
                g_loss.append(G_loss_curr)

                print('%d/%d' % (it, CONFIG.epochs), '|    D_loss:', d_loss[-1], '    G_loss:', g_loss[-1],
                      '  %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

                if it % CONFIG.per_save == 0:
                    samples = mydcgan.predict(sess, Data.sample_Z(25, CONFIG.noise_dim))
                    Plot.plot_image(np.clip((samples + 1) * 127.5, 0, 255).astype(np.uint8), str(it))

                    mydcgan.save(sess, CONFIG.model_save_path)
            print('训练结束时间：  %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            Plot.plot_loss(g_loss, d_loss)

    @staticmethod
    def predict():
        with tf.Session() as sess:
            mydcgan = DCGAN(CONFIG)
            mydcgan.restore(sess, CONFIG.model_save_path)

            samples = mydcgan.predict(sess, Data.sample_Z(25, CONFIG.noise_dim))
            Plot.plot_image(np.clip((samples + 1) * 127.5, 0, 255).astype(np.uint8))


def main(unused_argvs):
    if CONFIG.mode.startswith('train'):
        User.train()
    elif CONFIG.mode == 'predict':
        User.predict()


if __name__ == '__main__':
    tf.app.run()
