import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.contrib import layers as l
from tqdm import tqdm
import os, sys, glob
from collections import OrderedDict

#### Default parameters ################
tf.app.flags.DEFINE_integer("batch_size", 64, "mini batch training size")
tf.app.flags.DEFINE_float("critic_lr", 1e-3, "critic learning rate")
tf.app.flags.DEFINE_integer("num_epochs", 10, "number of training epochs")
tf.app.flags.DEFINE_integer("im_size", 32, "input image size")
tf.app.flags.DEFINE_integer("z_dim", 256, "input noise dimension")
config=tf.app.flags.FLAGS
config(sys.argv, known_only=True)

class PGAN:
    def __init__(self, d_path, save_path):
        self.d_path = d_path
        self.save_path = save_path
        self.weight_init = tf.random_normal_initializer()
        self.activation = tf.nn.leaky_relu
        self.num_filters = 512
        self.init_size = 4
    def gen_input(self, z):
        """
        creates initial layers of the generator
        Inputs: noise (z) sampled from normal distribution (points in a d-dimensional hypersphere)
        Output: an image of size (batch_size, init_size, init_size, num_filters)
        """
        num_h = (self.init_size**2)*self.num_filters
        z = tf.nn.l2_normalize(z, axis=-1)
        x = l.fully_connected(z, num_h, activation_fn=self.activation,\
            weights_initializer=self.weight_init, scope='g_layer_0_fc')
        x = tf.reshape(x, shape=(-1, self.init_size, self.init_size, self.num_filters))
        x = l.conv2d(x, self.num_filters, 3, activation_fn=self.activation,\
            weights_initializer=self.weight_init, scope='g_layer_0_conv')
        return x
    def gen_output(self, x, name=''):
        """
        append output layer to the generator graph
        Input: input image(x) of size NHWC
        Output: RGB image of size NHW3
        N--batch size, H---image height, W---image width
        """
        x = l.conv2d(x, 3, 1, activation_fn=None, weights_initializer=self.weight_init, scope='g_output'+name)
        return x
    def main_block(self, x, num_filters, name):
        """
        pgan main block
        Inputs: output of the previous layer (x)
                a list (num_filters) with two elements representing the number of filters for the first and
                second convolutions.
                scope name for the opreations(name)
        """
        x  = l.conv2d(x, num_filters[0], 3, activation_fn=self.activation,\
            weights_initializer=self.weight_init, scope=name+'_conv_1')
        x = l.conv2d(x, num_filters[1], 3, activation_fn=self.activation,\
            weights_initializer=self.weight_init, scope=name+'_conv_2')
        return x
    def generator(self, z, alpha, d, graph=None, filters=None, reuse=False, name=''):
        """
        Creates the generator network
        Inputa: z---noise vector with random points from a hypersphere
                alpha-------ratio to combine current and previous output images
                graph------- computational graph ????????????
                filters------ a list with two elements(number of filters for the new layer)
                reuse------- create a copy of the network
        Output: N - RGB images of size HW
        """
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            if graph:
                x = graph.get_tensor_by_name(name)
                print(x.shape)
                x_size = x.get_shape().as_list()[1]
                x_upsample = tf.stop_gradient(tf.image.resize_nearest_neighbor(x, size=(x_size*2, x_size*2)))
                x = self.main_block(x_upsample, name='g_layer_%d'%(d), num_filters=filters)
                out_curr = self.gen_output(x, 'curr')
                out_prev = self.gen_output(x_upsample, 'prev')
                out = (1-alpha)*out_prev + alpha*out_curr
            else:
                x = self.gen_input(z)
                out = self.gen_output(x)
        return out, x.name

    def discriminator(self, ):
        print("to be done")
    def train(self, kernels):
        network = self.get_filters(kernels)
        for i, k in enumerate(network.values(), 1):
            tf.reset_default_graph()
            sess = tf.Session()
            graph = tf.get_default_graph()
            if i>1:
                p = self.save_path + 'checkpoint/%d_layers/'%(i-1)
                prev_graph = tf.train.import_meta_graph(p + 'partial.meta')
                prev_graph.restore(sess, tf.train.latest_checkpoint(p))
            with tf.name_scope('input_place_hoders'):
                x = tf.placeholder(tf.float32, shape=(None, self.init_size*(2**i), self.init_size*(2**i), 3), name='input_image')
                z = tf.placeholder(tf.float32, shape=(None, config.z_dim), name='latent_vector')
                alpha = tf.placeholder(tf.float32, shape=(), name='smoothing_ratio')
            with tf.name_scope('generator'):
                if i>1:
                    gen, tensor_name = self.generator(z, alpha, i, graph, filters=k[-1], name=tensor_name)
                else:
                    gen, tensor_name = self.generator(z, alpha, i)

            g_vars = [w for w in tf.trainable_variables() if 'generator' in w.name]
            print('layer: ', i, gen.shape)
            for op in g_vars:
                print(op)
            with tf.name_scope('graph_saver'):
                ckpt_path = self.save_path + 'checkpoint/%d_layers/'%(i)
                if not os.path.isdir(ckpt_path):
                    os.makedirs(ckpt_path)
                saver = tf.train.Saver([w for w in g_vars if not 'g_output' in w.name])

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            tf.get_default_graph().finalize()
            with tf.Session() as sess:
                sess.run(init_op)
                saver.save(sess, ckpt_path + 'partial')
        print("to be done")
    def minibatchstdev(self,):
        print("to be done")
    def load_data(self, ):
        print("to be done")
    def pixel_norm(self, ):
        print("to be done")
    def get_filters(self, kernels):
        filters = dict()
        for i, k in enumerate(kernels, 1):
            a = np.array(kernels[:i])
            filters['network_%d'%(i)] = np.stack((a, a//2), axis=1)
        filters = OrderedDict(sorted(filters.items(), key=lambda v: len(v[1])))
        return filters
