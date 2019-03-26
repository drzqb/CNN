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

        self.input()
        self.Loss_Optim()

    def input(self):
        self.X = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL],
                                name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.config.noise_dim], name='Z')

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

    def discriminator(self, x,is_train=True,reuse=False):
        with tf.variable_scope('discriminator',reuse=reuse):
            out_1 = tf.layers.Conv2D(32,4,2,'same')(x)
            out_1=tf.layers.Flatten()(out_1)
            out_2=tf.nn.leaky_relu(out_1)

            out_3 = tf.layers.Dense(100)(out_2)
            out_3=tf.layers.batch_normalization(out_3,axis=0,epsilon=EPS,training=is_train)
            out_4 = tf.nn.leaky_relu(out_3)

            out_5 = tf.layers.Dense(1,activation='sigmoid')(out_4)

            return out_5


    def generator(self, noise,is_train=True,reuse=False):
        with tf.variable_scope('generator',reuse=reuse):
            out_1=tf.layers.Dense(8 * IMAGE_SIZE * IMAGE_SIZE)(noise)
            out_1 = tf.reshape(out_1, [-1, IMAGE_SIZE // 2, IMAGE_SIZE // 2, 32])
            out_1=tf.layers.batch_normalization(out_1,axis=[0,1,2],epsilon=EPS,training=is_train)
            out_1 = tf.nn.relu(out_1)

            out_2=tf.layers.Conv2DTranspose(IMAGE_CHANNEL,5,2,'same',activation='tanh')(out_1)

            return out_2

    def Loss_Optim(self):
        self.fake_image = self.generator(self.Z)
        self.fake_image_pre = self.generator(self.Z,is_train=False,reuse=True)
        D_fake = self.discriminator(self.fake_image)
        D_real = self.discriminator(self.X,reuse=True)

        # loss includes 'LS' and 'Log'
        if self.config.loss_mode is None or self.config.loss_mode == 'LS':
            self.D_loss = tf.reduce_mean((D_real - 1.0) ** 2 + D_fake ** 2)
            self.G_loss = tf.reduce_mean((D_fake - 1.0) ** 2)
        else:
            self.D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
            self.G_loss = -tf.reduce_mean(tf.log(D_fake))

        var_D=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
        var_G=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
        self.D_train_op = tf.train.AdamOptimizer(self.config.lr_d, beta1=self.config.beta_d).minimize(self.D_loss,
                                                                                                      var_list=var_D)
        self.G_train_op = tf.train.AdamOptimizer(self.config.lr_g, beta1=self.config.beta_g).minimize(self.G_loss,
                                                                                                      var_list=var_G)


class Data():
    @staticmethod
    def sample_Z(m, n):
        return np.random.uniform(-1., 1., size=[m, n])


class Plot():
    @staticmethod
    def plot_image(samples, suffix=None):
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
