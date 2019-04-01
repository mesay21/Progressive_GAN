import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib import layers as l
from tqdm import tqdm
import numpy as np

tf.app.flags.DEFINE_integer("batch_size", 64, "mini batch training size")
tf.app.flags.DEFINE_integer("epoch", 100, "number of training epochs")
tf.app.flags.DEFINE_float("critic_lr", 1e-3, "critic learning rate")
tf.app.flags.DEFINE_float('gen_lr', 1e-3, "Generator learning rate")
tf.app.flags.DEFINE_integer('im_size', 32, "input image size")
config=tf.app.flags.FLAGS
config(sys.argv, known_only=True)

class PGAN:
    def __init__(self, d_path, save_path):
        ####initial variables to be defined
        self.d_path = d_path
        self.save_path = save_path
        print('Network initialized')

    def generator_init_block(self, inputs, out_size, num_kernels):
        num_h = (out_size**2)*num_kernels
        x = l.fully_connected(inputs, num_h, activation_fn=tf.nn.leaky_relu, name='g_fc_layer')
        x = tf.reshape(x, (-1, out_size, out_size, num_kernels))
        x = l.conv2d(x, num_kernels, 3, activation_fn.tf.nn.leaky_relu, name='g_conv_init')
        return x
    def main_block(self, inputs, num_kernels, d, op, ntk):
        if ntk=='upsample':
            size = tf.shape(inputs)
            inputs = tf.image.resize_nearest_neighbor(inputs, (size[0]*2, size[1]*2))
        x = input
        for i, k in enumerate(num_kernels):
            x = l.conv2d(x, k, 3, activation_fn=tf.nn.leaky_relu, name='%s_conv_%d_%d'%(ntk, d, i))
        if ntk=='downsample':
            x = l.avg_pool2d(x, 2, name='%s_pool_%d_1'%(ntk, d))
        return x
    def critic_final_block(self, x, num_kernels):
        x = minibatchstddev(x)
        x = l.conv2d(x, num_kernels, 3, activation_fn=tf.nn.leaky_relu, name='d_init_1')
        x = l.conv2d(x, num_kernels, 4, activation_fn=tf.nn.leaky_relu, name='d_inint_2', padding='VALID')
        x = l.flatten(x, name='flatten')
        x = l.fully_connected(x, 1, activation_fn=None, name='output')
        return x

    def generator(self, z_dim, init_size, alpha, layers=None, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            z =  tf.nn.l2_normalize(tf.random_normal(shape=(None, z_dim)), axis=1)
            x = self.generator_init_block(z, init_size[0], init_size[1])
            if layers is not None:
                for i, k in enumerate(layers, 1):
                    if i==len(layers):
                        x_prev = x
                    x = self.main_block(x, k, i, 'upsample', 'g')

                out = l.conv2d(x, 3, 1, name='output', activation_fn=tf.nn.tanh)
                size = tf.shape(x_prev)
                out_prev = tf.image.resize_nearest_neighbor(x_prev, (size[0]*2, size[1]*2))
                out_prev = l.conv2d(x_prev, 3, 1, name='prev_out', activation_fn=tf.nn.tanh)
                out = (1 - alpha)*out_prev + alpha*out
                return out
            else:
                out = l.conv2d(x, 3, 1, name='output', activation_fn=tf.nn.tanh)
                return out

    def discrminator(self, inputs, num_kernels, alpha, layers=None, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            x = l.conv2d(inputs, num_kernels, 1, name='d_input_conv', activation_fn=tf.nn.leaky_relu)
            if layers is not None:
                for i, k in enumerate(layers, 1):
                    if i==0:
                        size = tf.shape(inputs)
                        inputs = tf.image.resize_nearest_neighbor(inputs, (size[0]//2, size[1]//2))
                        x_prev = l.conv2d(inputs, num_kernels, 1, name='d_smooth-trans', activation_fn=tf.nn.leaky_relu)
                        x = self.main_block(x, k, i, 'downsample', 'd')
                        assert x.shape==x_prev.shape
                        x = (1-alpha)*x + alpha*x_prev
                    else:
                        x = main_block(x, k, i, 'downsample', 'd')

            x = self.critic_final_block(x, num_kernels)
            return x
    def minibatchstddev(self, x):
        x_shape = tf.shape(x)
        _, x_std = tf.nn.moments(x, axes=0)
        x_mean, _ = tf.nn.moments(x_std, axes=[0, 1, 2], keep_dims=True)
        std_stat = tf.tile(x_mean, x_shape[:-1])
        x = tf.concat((x, std_stat), axis=-1)
        assert (x_shape[-1]+1)==x.shape[-1]
        return x

    def train(self, init_size, im_size, kernel, save_path):
        images, _ = self.load_data(d_path)
        kernel_list = get_filters(kernel)
        for i, k, v in enumerate(kernel_list.items(), 1):
            print('Started training %d layers'%(i))
            tf.reset_default_graph()
            #### Data reading #####################
            data = tf.placeholder(tf.float32, shape=(None, config.image_size, config.image_size, 3))
            x_read = tf.data.Dataset.from_tensor_slices(data)
            x_read = x_read.shuffle(10000)
            x_read = x_read.batch(config.batch_size)
            x_iterator = x_read.make_initializable_iterator()
            x_next = x_iterator.get_next()
            #### Create place holders #############
            x = tf.placeholder(tf.float32, shape=(None, im_size, im_size, 3), name='input_image')
            alpha = tf.placeholder(tf.float32, shape=(), name='contribution_ratio')
            beta = tf.placeholder(tf.float32, shape=(), name='gp_ratio')
            gamma = tf.placeholder(tf.float32, shape=(), name='gamma_value')
            lambda_ = tf.placeholder(tf.float32, shape=(), name='gp_contribution')

            #### Create the network   #############
            with tf.name_scope('generator'):
                g = self.generator(z_dim, [init_size, v[0]], layers=v[1:])
            with tf.name_scope('discriminator'):
                d_filters = v.reverse()
                print('checking if filters are reversed', v, d_filters)
                d_real = self.discriminator(x, v[-1], alpha, layers=d_filters[1:])
                d_fake = self.discriminator(g, v[-1], alpha, layers=d_filters[1:], reuse=True)
            with tf.name_scope('Loss'):
                eps = tf.random_uniform(shape=tf.shape(x))
                x_hat = (1-eps)*x + eps*g
                d_hat = self.discriminator(x_hat, v[-1], alpha, layers=d_filters[1:], reuse=True)
                d_grad = tf.gradients(d_hat, x_hat)[0]
                print('gradient shape', d_grad.shape)
                d_grad = tf.sqrt(tf.reduce_sum(tf.square(d_grad), axis=1))
                print('gradient shape after magnitude', d_grad.shape)
                gp = tf.reduce_mean(((d_grad - gamma)**2)/(gamma**2))
                d_loss = tf.reduce_mean(d_fake)-tf.reduce_mean(d_real) + lambda_*gp
                g_loss = -tf.reduce_mean(d_fake)
            g_vars = [v for v in tf.trainable_variables if 'generator' in v.name]
            d_vars = [v for v in tf.trainable_variables if 'discriminator' in v.name]
            with tf.name_scope('optimization'):
                g_optim = tf.train.AdamOptimizer(0.001, 0.0, 0.99, name='g_optimizer').minimize(g_loss, var_list=g_vars)
                d_optim = tf.train.AdamOptimizer(0.001, 0.0, 0.99, name='d_optimizer').minimize(d_loss, var_list=d_vars)
            with tf.name_scope('summary'):
                d_loss_summ = tf.summary.scalar('critic_loss', d_loss)
                g_loss_summ = tf.summary.scalar('generator_loss', g_loss)
                summary = tf.summary.merge_all()
                summ_logdir = save_path + '/summary/'
                chkpt_logdir = save_path + '/checkpoint/%d_layers.ckpt'%(i)
                summ_writer = tf.summary.FileWriter(summ_logdir)
                saver = tf.train.Saver()
                if i>1:
                    restore_dir = save_path + '/checkpoint/%d_layers.ckpt'%(i-1)
                    prev_ntk = [v for v in g_vars if not 'g_conv_%d'%(i) in v.name] +\
                        [v for v in d_vars if not 'd_conv_%d'%(i) in v.name]
                    prev_ntk_saver = tf.train.Saver(prev_ntk)

                init_op = tf.group(tf.global_variables(), tf.local_variables())
            with tf.Session() as sess:
                sess.run(x_iterator.initializer, feed_dict={data:images})
                prev_ntk_saver.restore(sess, restore_dir)
                for v in init_op:
                    if not v.is_variable_initialized():
                        sess.run(v.initializer)
                summ_writer.add_graph(sess.graph)
                ####### reading data is to be done#######
                for i in tqdm(range(num_epochs)):
                    ###### run the networks
                    print('to be done')
                    ###### write summary#######
                    if i%5==1 or i==config.num_epochs:
                        ########summary
                        print('to be done')


            print('Finished training %d layers'%(i))

    def get_filters(self, kernel_list):
        filters = dict()
        for i, k in enumerate(kernel_list, 1):
            filters['network_%d'%(i)]=kernel_list[:i]
        return filters

    def load_data(self, path, im_size):
        f_names = glob.glob(path)
        data, labels = []
        for l in f_names:
            with open(l, 'rb') as f:
                x = pickle.load(f, encoding='bytes')
            data.append(x[b'data'].reshpe((-1, *im_size), order='F').swapaxes(2, 1))
            labels.extend(x[b'labels'])
        data = np.concatinate(data)
        print('training set size', data.shape)
        assert(len(labels)==len(data))
        return data, labels


