import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib import layers as l
from tqdm import tqdm
import numpy as np
import pickle
import sys
import glob
from collections import OrderedDict

tf.app.flags.DEFINE_integer("batch_size", 64, "mini batch training size")
tf.app.flags.DEFINE_integer("epoch", 2, "number of training epochs")
tf.app.flags.DEFINE_float("critic_lr", 1e-3, "critic learning rate")
tf.app.flags.DEFINE_float('gen_lr', 1e-3, "Generator learning rate")
tf.app.flags.DEFINE_integer('im_size', 32, "input image size")
tf.app.flags.DEFINE_integer('z_dim', 256, "input noise dimensionality")
config=tf.app.flags.FLAGS
config(sys.argv, known_only=True)

class PGAN:
    def __init__(self, d_path, save_path):
        ####initial variables to be defined
        self.d_path = d_path
        self.save_path = save_path
        print('Network initialized')

    def generator_init_block(self, inputs, out_size, num_kernels):
        ##### first layer of the generator #######
        num_h = (out_size**2)*num_kernels
        x = l.fully_connected(inputs, num_h, activation_fn=tf.nn.leaky_relu)
        x = tf.reshape(x, (-1, out_size, out_size, num_kernels))
        x = l.conv2d(x, num_kernels, 3, activation_fn=tf.nn.leaky_relu)
        return x
    def main_block(self, inputs, num_kernels, op):
        ##### takes an input and a list with the number of kernels for each layer #######
        if op=='upsample':
            size = inputs.get_shape()
            inputs = tf.image.resize_nearest_neighbor(inputs, (size[1]*2, size[1]*2))
        x = inputs
        x = l.conv2d(x, num_kernels, 3, activation_fn=tf.nn.leaky_relu)
        if op=='downsample':
            x = l.avg_pool2d(x, 2)
        return x
    def critic_final_block(self, x, num_kernels):
        x = self.minibatchstddev(x)
        x = l.conv2d(x, num_kernels, 3, activation_fn=tf.nn.leaky_relu)
        x = l.conv2d(x, num_kernels, 4, activation_fn=tf.nn.leaky_relu, padding='VALID')
        x = l.flatten(x)
        x = l.fully_connected(x, 1, activation_fn=None)
        return x

    def generator(self, z, init_size, alpha, layers=None, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            z =  tf.nn.l2_normalize(z, axis=1)
            x = self.generator_init_block(z, init_size[0], init_size[1])
            if layers:
                for i, k in enumerate(layers, 1):
                    if i==len(layers):
                        x_prev = x
                    x = self.main_block(x, k, 'upsample')
                out = l.conv2d(x, 3, 1, activation_fn=tf.nn.tanh)
                size = out.get_shape()
                out_prev = tf.image.resize_nearest_neighbor(x_prev, size=(size[1], size[1]))
                out_prev = l.conv2d(out_prev, 3, 1, activation_fn=tf.nn.tanh)
                out = (1 - alpha)*out_prev + alpha*out
                return out
            else:
                out = l.conv2d(x, 3, 1, activation_fn=tf.nn.tanh)
                return out

    def discriminator(self, inputs, num_kernels, in_size, alpha, layers=None, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            x = l.conv2d(inputs, num_kernels, 1, activation_fn=tf.nn.leaky_relu)
            if layers:
                for i, k in enumerate(layers, 1):
                    if i==0:
                        inputs = tf.image.resize_nearest_neighbor(inputs, (in_size//2, in_size//2))
                        x_prev = l.conv2d(inputs, num_kernels, 1, activation_fn=tf.nn.leaky_relu)
                        x = self.main_block(x, k, 'downsample')
                        assert x.shape==x_prev.shape
                        x = (1-alpha)*x + alpha*x_prev
                    else:
                        x = self.main_block(x, k, 'downsample')

            x = self.critic_final_block(x, num_kernels)
            return x
    def minibatchstddev(self, x):
        _, x_std = tf.nn.moments(x, axes=0)
        x_mean, _ = tf.nn.moments(x_std, axes=[0, 1, 2], keep_dims=True)
        std_stat = tf.expand_dims(tf.tile(x_mean, tf.shape(x)[:-1]), axis=-1)
        x = tf.concat((x, std_stat), axis=-1)
        return x

    def train(self, init_size, kernel):
        images, _ = self.load_data(self.d_path, [config.im_size, config.im_size, 3])
        images = 2*(images/255)-1
        print(np.min(images), np.max(images))
        kernel_list = self.get_filters(kernel)
        for i, v in enumerate(kernel_list.values(), 1):
            print('Started training %d layers'%(i))
            tf.reset_default_graph()
            #### Data reading #####################
            data = tf.placeholder(tf.float32, shape=(None, config.im_size, config.im_size, 3))
            x_read = tf.data.Dataset.from_tensor_slices(data)
            x_read = x_read.shuffle(10000)
            x_read = x_read.batch(config.batch_size)
            x_iterator = x_read.make_initializable_iterator()
            x_next = x_iterator.get_next()
            #### Create place holders #############
            x = tf.placeholder(tf.float32, shape=(None, init_size*i, init_size*i, 3), name='input_image')
            z = tf.placeholder(tf.float32, shape=(None, config.z_dim), name='noise_input')
            alpha = tf.placeholder(tf.float32, shape=(), name='contribution_ratio')
            beta = tf.placeholder(tf.float32, shape=(), name='gp_ratio')
            gamma = tf.placeholder(tf.float32, shape=(), name='gamma_value')
            lambda_ = tf.placeholder(tf.float32, shape=(), name='gp_contribution')
            im_resize = tf.image.resize_nearest_neighbor(data, size=(init_size*i, init_size*i))

            #### Create the network   #############
            with tf.name_scope('generator'):
                g = self.generator(z, [init_size, v[0]], layers=v[1:], alpha=alpha)
                out_shape = g.get_shape()[1]
            with tf.name_scope('discriminator'):
                v.reverse()
                d_real = self.discriminator(x, v[-1], out_shape, alpha, layers=v[1:])
                d_fake = self.discriminator(g, v[-1], out_shape, alpha, layers=v[1:], reuse=True)
            with tf.name_scope('Loss'):
                eps = tf.random_uniform(shape=tf.shape(g))
                x_hat = (1-eps)*x + eps*g
                d_hat = self.discriminator(x_hat, v[-1], out_shape, alpha, layers=v[1:], reuse=True)
                d_grad = tf.gradients(d_hat, x_hat)[0]
                d_grad = tf.sqrt(tf.reduce_sum(tf.square(d_grad), axis=(1, 2, 3)))
                gp = tf.reduce_mean(((d_grad - gamma)**2)/(gamma**2))
                d_loss = tf.reduce_mean(d_fake)-tf.reduce_mean(d_real) + lambda_*gp
                g_loss = -tf.reduce_mean(d_fake)
            g_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]
            d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]
            with tf.name_scope('optimization'):
                g_optim = tf.train.AdamOptimizer(0.001, 0.0, 0.99, name='g_optimizer').minimize(g_loss, var_list=g_vars)
                d_optim = tf.train.AdamOptimizer(0.001, 0.0, 0.99, name='d_optimizer').minimize(d_loss, var_list=d_vars)
            with tf.name_scope('summary'):
                d_loss_summ = tf.summary.scalar('critic_loss', d_loss)
                g_loss_summ = tf.summary.scalar('generator_loss', g_loss)
                summary = tf.summary.merge_all()
                summ_logdir = self.save_path + '/summary/'
                chkpt_logdir = self.save_path + '/checkpoint/%d_layers.ckpt'%(i)
                summ_writer = tf.summary.FileWriter(summ_logdir)
                saver = tf.train.Saver()
                if i>1:
                    restore_dir = save_path + '/checkpoint/%d_layers.ckpt'%(i-1)
                    prev_ntk = [v for v in g_vars if not 'g_conv_%d'%(i) in v.name] +\
                        [v for v in d_vars if not 'd_conv_%d'%(i) in v.name]
                    prev_ntk_saver = tf.train.Saver(prev_ntk)

                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            with tf.Session() as sess:
                sess.run(x_iterator.initializer, feed_dict={data:images})
                sess.run(init_op)
                if i>1:
                    prev_ntk_saver.restore(sess, restore_dir)
                summ_writer.add_graph(sess.graph)
                ####### reading data is to be done#######
                for a in tqdm(range(config.epoch)):
                    ###### run the networks #############
                    for b in range(10):
                        ratio = 1-(1/np.exp(b))
                        noise = np.random.normal(size=(config.batch_size, config.z_dim))
                        x_batch = sess.run(x_next)
                        x_batch = im_resize.eval(feed_dict={data:x_batch})
                        sess.run(d_optim, feed_dict={x:x_batch, z:noise, alpha:ratio, lambda_:10, gamma:750})
                        sess.run(g_optim, feed_dict={x:x_batch, z:noise, alpha:ratio, lambda_:10, gamma:750})
                    ###### write summary#######
                    if a%5==1 or a==config.epoch:
                        ######## summary #########
                        gen_images = g.eval(feed_dict={z:noise, alpha:ratio})
                        summ = sess.run(summary, feed_dict={x:x_batch, z:noise, alpha:ratio, lambda_:10, gamma:750})
                        summ_writer.add_summary(summ, i)
                saver.save(sess, chkpt_logdir)
            print('Finished training %d layers'%(i))

    def get_filters(self, kernel_list):
        filters = dict()
        for i, k in enumerate(kernel_list, 1):
            filters['network_%d'%(i)]=kernel_list[:i]
        filters = OrderedDict(sorted(filters.items(), key=lambda v: v[1]))
        return filters

    def load_data(self, path, im_size):
        f_names = glob.glob(path)
        data, labels = [], []
        for l in f_names:
            with open(l, 'rb') as f:
                x = pickle.load(f, encoding='bytes')
            data.append(x[b'data'].reshape((-1, *im_size), order='F').swapaxes(2, 1))
            labels.extend(x[b'labels'])
        data = np.concatenate(data)
        assert(len(labels)==len(data))
        return data, labels
def save_images(x, fname):
    s = x.shape
    x = np.concatenate([v.reshape(-1, s[-2], s[-1]) for v in np.split(x, 8, axis=0)], axis=1)
    x = 255*(0.5*x + 0.5)
    x = x.astype(np.uint8)
    plt.imsave(fname, x)


