import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib import layers as l
from tqdm import tqdm
import numpy as np
import sys, glob, os, pickle
from collections import OrderedDict

tf.app.flags.DEFINE_integer("batch_size", 64, "mini batch training size")
tf.app.flags.DEFINE_integer("epoch", 5, "number of training epochs")
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

    def generator_init_block(self, inputs, out_size, num_kernels, var_scope):
        init = tf.random_normal_initializer()
        ##### first layer of the generator #######
        num_h = (out_size**2)*num_kernels
        x = l.fully_connected(inputs, num_h, activation_fn=tf.nn.leaky_relu,\
             scope=var_scope +'_fc', weights_initializer=init)
        x = tf.reshape(x, (-1, out_size, out_size, num_kernels))
        x = l.conv2d(x, num_kernels, 3, activation_fn=tf.nn.leaky_relu,\
             scope=var_scope + '_conv', weights_initializer=init)
        x = self.pixel_normalization(x)
        return x
    def main_block(self, inputs, num_kernels, op, var_scope):
        init = tf.random_normal_initializer()
         ##### takes an input and a list with the number of kernels for each layer #######
        if op=='upsample':
            size = inputs.get_shape()
            inputs = tf.image.resize_nearest_neighbor(inputs, (size[1]*2, size[1]*2))
            x = l.conv2d(inputs, num_kernels, 3, activation_fn=tf.nn.leaky_relu,\
                 scope=var_scope+'_conv_1', weights_initializer=init)
            x = self.pixel_normalization(x)
            x = l.conv2d(x, num_kernels//2, 3, activation_fn=tf.nn.leaky_relu,\
                 scope=var_scope+'_conv_2', weights_initializer=init)
            x = self.pixel_normalization(x)
        if op=='downsample':
            x = l.conv2d(inputs, num_kernels, 3, activation_fn=tf.nn.leaky_relu,\
                 scope=var_scope+'_conv_1', weights_initializer=init)
            x = self.pixel_normalization(x)
            x = l.conv2d(x, num_kernels*2, 3, activation_fn=tf.nn.leaky_relu,\
                 scope=var_scope+'_conv_2', weights_initializer=init)
            x = self.pixel_normalization(x)
            x = l.avg_pool2d(x, 2, scope=var_scope+'_pool')
        return x
    def critic_final_block(self, x, num_kernels, var_scope):
        init = tf.random_normal_initializer()
        x = self.minibatchstddev(x)
        x = l.conv2d(x, num_kernels, 3, activation_fn=tf.nn.leaky_relu,\
             scope=var_scope+'_conv_1', weights_initializer=init)
        x = self.pixel_normalization(x)
        x = l.conv2d(x, num_kernels, 4, activation_fn=tf.nn.leaky_relu,\
             padding='VALID', scope=var_scope+'_conv_2', weights_initializer=init)
        x = self.pixel_normalization(x)
        x = l.flatten(x, scope=var_scope)
        x = l.fully_connected(x, 1, activation_fn=None,\
             scope=var_scope+'_fc', weights_initializer=init)
        return x

    def generator(self, z, init_size, alpha, layers=None, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            init = tf.random_normal_initializer()
            z =  tf.nn.l2_normalize(z, axis=1)
            x = self.generator_init_block(z, init_size[0], init_size[1], 'g_layer_0')
            if layers:
                for i, k in enumerate(layers, 1):
                    if i==len(layers):
                        x_prev = x
                    x = self.main_block(x, k, 'upsample', 'g_layer_%d'%(i))
                out = l.conv2d(x, 3, 1, activation_fn=tf.nn.tanh,\
                     scope='g_output_1', weights_initializer=init)
                size = out.get_shape()
                ####### may be not correct #########
                out_prev = tf.stop_gradient(tf.image.resize_nearest_neighbor(x_prev, size=(size[1], size[1])))
                out_prev = l.conv2d(out_prev, 3, 1, activation_fn=tf.nn.tanh,\
                     scope='g_output_0', weights_initializer=init)
                out = (1 - alpha)*out_prev + alpha*out
                return out
            else:
                out = l.conv2d(x, 3, 1, activation_fn=tf.nn.tanh,\
                     scope='g_output_1', weights_initializer=init)
                return out

    def discriminator(self, inputs, in_kernel, out_kernel, in_size, alpha, layers=None, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            init = tf.random_normal_initializer()
            x = l.conv2d(inputs, in_kernel, 1, activation_fn=tf.nn.leaky_relu,\
                 scope='d_input_conv_0', weights_initializer=init)
            if layers:
                for i, k in enumerate(layers, 1):

                    idx = (len(layers)-i) + 1
                    if i==1:
                        inputs = tf.stop_gradient(tf.image.resize_nearest_neighbor(inputs, (in_size//2, in_size//2)))
                        x_prev = l.conv2d(inputs, k*2, 1, activation_fn=tf.nn.leaky_relu,\
                             scope='d_input_conv_1', weights_initializer=init)
                        x_prev = self.pixel_normalization(x_prev)
                        x = self.main_block(x, k, 'downsample', 'd_layer_%d'%(idx))
                        x = (1-alpha)*x + alpha*x_prev
                    else:
                        x = self.main_block(x, k, 'downsample', 'd_layer_%d'%(idx))
            x = self.critic_final_block(x, out_kernel, 'd_layer_0')
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
        kernel_list = self.get_filters(kernel)
        partial_vars = dict()
        trained_weight = dict()
        for i, v in enumerate(kernel_list.values(), 1):
            print('Started training %d layers'%(i))
            tf.reset_default_graph()
            #### Data reading #####################
            data = tf.placeholder(tf.float32, shape=(None, config.im_size, config.im_size, 3))
            x_read = tf.data.Dataset.from_tensor_slices(data)
            x_read = x_read.repeat()
            x_read = x_read.shuffle(10000)
            x_read = x_read.batch(config.batch_size)
            x_iterator = x_read.make_initializable_iterator()
            x_next = x_iterator.get_next()
            #### Create place holders #############
            x = tf.placeholder(tf.float32, shape=(None, init_size*(2**(i-1)), init_size*(2**(i-1)), 3), name='input_image')
            z = tf.placeholder(tf.float32, shape=(None, config.z_dim), name='noise_input')
            alpha = tf.placeholder(tf.float32, shape=(), name='contribution_ratio')
            beta = tf.placeholder(tf.float32, shape=(), name='gp_ratio')
            gamma = tf.placeholder(tf.float32, shape=(), name='gamma_value')
            lambda_ = tf.placeholder(tf.float32, shape=(), name='gp_contribution')
            im_resize = tf.image.resize_nearest_neighbor(data, size=(init_size*(2**(i-1)), init_size*(2**(i-1))))

            #### Create the network   #############
            with tf.name_scope('generator'):
                g = self.generator(z, [init_size, v[0]], layers=v[1:], alpha=alpha)
                out_shape = g.get_shape()[1]
                print('generator output', g.shape)
            with tf.name_scope('discriminator'):
                v.reverse()
                d_real = self.discriminator(x, v[0], v[-1], out_shape, alpha, layers=v[:-1])
                d_fake = self.discriminator(g, v[0], v[-1], out_shape, alpha, layers=v[:-1], reuse=True)
            with tf.name_scope('Loss'):
                eps = tf.random_uniform(shape=tf.shape(g))
                x_hat = (1-eps)*x + eps*g
                d_hat = self.discriminator(x_hat, v[0], v[-1], out_shape, alpha, layers=v[:-1], reuse=True)
                d_grad = tf.gradients(d_hat, x_hat)[0]
                d_grad = tf.sqrt(tf.reduce_sum(tf.square(d_grad), axis=(1, 2, 3)))
                gp = tf.reduce_mean(((d_grad - gamma)**2)/(gamma**2))
                d_loss = tf.reduce_mean(d_fake)-tf.reduce_mean(d_real) + lambda_*gp
                g_loss = -tf.reduce_mean(d_fake)
            g_vars = [w for w in tf.trainable_variables() if 'generator' in w.name]
            d_vars = [w for w in tf.trainable_variables() if 'discriminator' in w.name]
            partial_vars['layer_%d'%(i)] = [w for w in g_vars if not 'g_output' in w.name] + [w for w in d_vars if not 'd_input' in w.name]
            with tf.name_scope('weight_clipping'):
                weights = [w for w in tf.trainable_variables() if 'weights' in w.name]
                weight_clip = tf.group([w/(tf.sqrt(2./tf.cast(tf.reduce_prod(w.shape[:-1]), tf.float32))) for w in weights])
            with tf.name_scope('optimization'):
                g_optim = tf.train.AdamOptimizer(0.001, 0.0, 0.99, name='g_optimizer').minimize(g_loss, var_list=g_vars)
                d_optim = tf.train.AdamOptimizer(0.001, 0.0, 0.99, name='d_optimizer').minimize(d_loss, var_list=d_vars)
            with tf.name_scope('summary'):
                d_loss_summ = tf.summary.scalar('critic_loss', d_loss)
                g_loss_summ = tf.summary.scalar('generator_loss', g_loss)
                summary = tf.summary.merge_all()
                summ_logdir = self.save_path + 'summary/'
                im_path = self.save_path + 'images/%d_layers/'%(i)
                full_model_logdir = self.save_path + 'checkpoint/%d_layers_model.ckpt'%(i)
                partial_model_logdir = self.save_path + 'checkpoint/%d_partial_model.ckpt'%(i)
                summ_writer = tf.summary.FileWriter(summ_logdir)
            with tf.name_scope('save_operation'):
                full_model = tf.train.Saver(name='full_model_saver')
                partial_model = tf.train.Saver(partial_vars['layer_%d'%(i)], name='partail_model_saver')
                if not os.path.isdir(im_path):
                    os.makedirs(im_path)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            with tf.Session() as sess:
                sess.run(init_op)
                if i>1:
                    for w in tf.global_variables():
                        if w.name in trained_weight.keys():
                            w.load(trained_weight[w.name])
                summ_writer.add_graph(sess.graph)
                sess.run(x_iterator.initializer, feed_dict={data:images})
                ####### reading data is to be done#######
                for a in tqdm(range(config.epoch)):
                    ###### run the networks #############
                    num_minibatches = len(images)//config.batch_size
                    for b in range(num_minibatches):
                        ratio = 1-(1/np.exp(b/num_minibatches))
                        print
                        noise = np.random.normal(size=(config.batch_size, config.z_dim))
                        x_batch = sess.run(x_next)
                        x_batch = im_resize.eval(feed_dict={data:x_batch})
                        sess.run(d_optim, feed_dict={x:x_batch, z:noise, alpha:ratio, lambda_:10, gamma:750})
                        sess.run(g_optim, feed_dict={x:x_batch, z:noise, alpha:ratio, lambda_:10, gamma:750})
                        sess.run(weight_clip)
                    ###### write summary#######
                    # if a%5==0 or a==(config.epoch-1):
                ######## summary #########
                    gen_images = g.eval(feed_dict={z:noise, alpha:ratio})
                    self.save_images(gen_images, im_path+'gen_%d.png'%(a))
                    self.save_images(x_batch, im_path+'real_%d.png'%(a))
                    summ = sess.run(summary, feed_dict={x:x_batch, z:noise, alpha:ratio, lambda_:10, gamma:750})
                    summ_writer.add_summary(summ, i)
                full_model.save(sess, full_model_logdir)
                partial_model.save(sess, partial_model_logdir, strip_default_attrs=True, write_meta_graph=False)
                for w in partial_vars['layer_%d'%(i)]:
                    trained_weight[w.name] = sess.run(w.value())
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
    def save_images(self, x, fname):
        s = x.shape
        x = np.concatenate([v.reshape(-1, s[-2], s[-1]) for v in np.split(x, 8, axis=0)], axis=1)
        x = 255*(0.5*x + 0.5)
        x = x.astype(np.uint8)
        plt.imsave(fname, x)
    def pixel_normalization(self, x):
        x_ratio = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8)
        return x/x_ratio



