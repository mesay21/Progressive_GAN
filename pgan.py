import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib import layers as l
from tqdm import tqdm
import numpy as np
import sys, glob, os, pickle
from collections import OrderedDict

tf.app.flags.DEFINE_integer("batch_size", 64, "mini batch training size")
tf.app.flags.DEFINE_integer("epoch", 100, "number of training epochs")
tf.app.flags.DEFINE_float("critic_lr", 1e-3, "critic learning rate")
tf.app.flags.DEFINE_float('gen_lr', 1e-3, "Generator learning rate")
tf.app.flags.DEFINE_integer('im_size', 32, "input image size")
tf.app.flags.DEFINE_integer('z_dim', 512, "input noise dimensionality")
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
        x = self.pixel_normalization(x)
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
            x = l.conv2d(x, num_kernels*2, 3, activation_fn=tf.nn.leaky_relu,\
                 scope=var_scope+'_conv_2', weights_initializer=init)
            x = l.avg_pool2d(x, 2, scope=var_scope+'_pool')
        return x
    def critic_final_block(self, x, num_kernels, var_scope):
        init = tf.random_normal_initializer()
        x = self.minibatchstddev(x)
        x = l.conv2d(x, num_kernels, 3, activation_fn=tf.nn.leaky_relu,\
             scope=var_scope+'_conv_1', weights_initializer=init)
        x = l.conv2d(x, num_kernels, 4, activation_fn=tf.nn.leaky_relu,\
             padding='VALID', scope=var_scope+'_conv_2', weights_initializer=init)
        x = l.flatten(x, scope=var_scope+'_flatten')
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
                out = l.conv2d(x, 3, 1, activation_fn=None,\
                     scope='g_output_1', weights_initializer=init)
                size = out.get_shape()
                ####### may be not correct #########
                out_prev = l.conv2d(x_prev, 3, 1, activation_fn=None,\
                     scope='g_output_0', weights_initializer=init)
                out_prev = tf.stop_gradient(tf.image.resize_nearest_neighbor(out_prev, size=(size[1], size[1])))
                out = out_prev + (out - out_prev)*tf.clip_by_value(alpha, 0.0, 1.0)
                return out
            else:
                out = l.conv2d(x, 3, 1, activation_fn=None,\
                     scope='g_output_1', weights_initializer=init)
            return out

    def discriminator(self, inputs, in_kernel, out_kernel, in_size, alpha, layers=None, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            init = tf.random_normal_initializer()
            image_prev = inputs
            x = l.conv2d(inputs, in_kernel, 1, activation_fn=tf.nn.leaky_relu,\
                 scope='d_input_conv_0', weights_initializer=init)
            if layers:
                for i, k in enumerate(layers, 1):

                    idx = (len(layers)-i) + 1
                    if i==1:
                        image_prev = l.avg_pool2d(image_prev, 2, scope='d_input_pool')
                        x_prev = l.conv2d(image_prev, k*2, 1, activation_fn=tf.nn.leaky_relu,\
                             scope='d_input_conv_1', weights_initializer=init)
                        x = self.main_block(x, k, 'downsample', 'd_layer_%d'%(idx))
                        x = x_prev + (x - x_prev)*tf.clip_by_value(alpha, 0.0, 1.0)
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
            alpha = tf.Variable(0.0, trainable=False, name='contribution_ratio')
            step = tf.placeholder(tf.float32, shape=(), name='step')
            trans = alpha.assign(2.*step/config.epoch, name='update_alpha')
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
                eps = tf.random_uniform(shape=tf.shape(g))
                x_hat = (1-eps)*x + eps*g
                d_hat = self.discriminator(x_hat, v[0], v[-1], out_shape, alpha, layers=v[:-1], reuse=True)
                d_grad = tf.gradients(d_hat, x_hat)[0]
                d_grad = tf.sqrt(tf.reduce_sum(tf.square(d_grad), axis=(1, 2, 3)))
                gp = tf.reduce_mean(((d_grad - gamma)**2)/(gamma**2))
            with tf.name_scope('Loss'):
                d_loss = tf.reduce_mean(d_fake)-tf.reduce_mean(d_real) + lambda_*gp + 0.001*tf.reduce_mean(tf.square(d_real))
                g_loss = -tf.reduce_mean(d_fake)
            g_vars = [w for w in tf.trainable_variables() if 'generator' in w.name]
            d_vars = [w for w in tf.trainable_variables() if 'discriminator' in w.name]
            partial_vars['layer_%d'%(i)] = [w for w in g_vars if not 'g_output' in w.name] + [w for w in d_vars if not 'd_input' in w.name]
            with tf.name_scope('weight_clipping'):
                weights = [w for w in tf.trainable_variables() if 'weights' in w.name]
                weight_clip = [tf.assign(w, w*tf.sqrt(2./tf.cast(tf.reduce_prod(w.shape[:-1]), tf.float32))) for w in weights]
            with tf.name_scope('optimization'):
                g_optim = tf.train.AdamOptimizer(0.001, 0.0, 0.99, name='g_optimizer').minimize(g_loss, var_list=g_vars)
                d_optim = tf.train.AdamOptimizer(0.001, 0.0, 0.99, name='d_optimizer').minimize(d_loss, var_list=d_vars)
            with tf.name_scope('summary'):
                d_loss_summ = tf.summary.scalar('critic_loss', -d_loss)
                g_loss_summ = tf.summary.scalar('generator_loss', -g_loss)
                d_real_hist = tf.summary.histogram('real_critic', d_real)
                d_fake_hist = tf.summary.histogram('fake_critic', d_fake)
                summary = tf.summary.merge_all()
                summ_logdir = self.save_path + 'summary/%d_layers/'%(i)
                im_path = self.save_path + 'images/%d_layers/'%(i)
                full_model_logdir = self.save_path + 'checkpoint/%d_layers_model.ckpt'%(i)
                partial_model_logdir = self.save_path + 'checkpoint/%d_partial_model.ckpt'%(i)
                summ_writer = tf.summary.FileWriter(summ_logdir)
            with tf.name_scope('save_operation'):
                full_model = tf.train.Saver(name='full_model_saver')
                partial_model = tf.train.Saver(partial_vars['layer_%d'%(i)], name='partail_model_saver')
                if not os.path.isdir(im_path):
                    os.makedirs(im_path)
            vars_ = tf.global_variables()
            init_vars = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            tf.get_default_graph().finalize()
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                sess.run(init_vars)
                sess.run(weight_clip)
                if i>1:
                    for w in vars_:
                        if w.name in trained_weight.keys():
                            w.load(trained_weight[w.name], sess)
                            assert (w.eval()==trained_weight[w.name]).all()
                summ_writer.add_graph(sess.graph)
                sess.run(x_iterator.initializer, feed_dict={data:images})
                for a in tqdm(range(config.epoch)):
                    ###### run the networks #############
                    num_minibatches = len(images)//config.batch_size
                    for _ in range(num_minibatches):###to be changed
                        # for w in weights:
                        #     print(w.name, np.min(w.eval()), np.max(w.eval()))
                        noise = np.random.normal(size=(config.batch_size, config.z_dim))
                        x_batch = sess.run(x_next)
                        # print(x_batch.shape, type(x_batch))
                        x_batch = im_resize.eval(feed_dict={data:x_batch})
                        # print(x_batch.shape, type(x_batch))
                        # gen_images = g.eval(feed_dict={z:noise})
                        sess.run(d_optim, feed_dict={x:x_batch, z:noise, lambda_:10., gamma:750.})
                        _, d_r, d_f = sess.run([g_optim, d_real, d_fake], feed_dict={x:x_batch, z:noise, lambda_:10., gamma:750.})
                        # if i>1:
                        #     print('real', d_r); print('fake', d_f)
                        #     print(np.min(x_batch), np.max(x_batch))
                        #     print(np.min(gen_images), np.max(gen_images))
                    sess.run(trans, feed_dict={step:np.float(a)})
                    ###### write summary#######
                    # if a%5==0 or a==(config.epoch-1):
                ######## summary #########
                    gen_images = np.clip(g.eval(feed_dict={z:noise}), -1., 1.)
                    self.save_images(gen_images, im_path+'gen_%d.png'%(a))
                    self.save_images(x_batch, im_path+'real_%d.png'%(a))
                    summ, d_l, g_l = sess.run([summary, d_loss, g_loss], feed_dict={x:x_batch, z:noise, lambda_:10, gamma:250})
                    summ_writer.add_summary(summ, a)
                    # print('Iteration: %d critic loss: %f generator loss %f'%(a, d_l, g_l))
                full_model.save(sess, full_model_logdir)
                partial_model.save(sess, partial_model_logdir, strip_default_attrs=True, write_meta_graph=False)
                for w in partial_vars['layer_%d'%(i)]:
                    trained_weight[w.name] = w.eval()
                    print(w.name)
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
        with tf.name_scope('pix_norm'):
            x_ratio = tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8)
            return x*x_ratio



