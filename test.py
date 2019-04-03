import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib import layers as l
from tqdm import tqdm
import numpy as np
import sys, glob, os, pickle
from collections import OrderedDict

tf.app.flags.DEFINE_integer("batch_size", 64, "mini batch training size")
tf.app.flags.DEFINE_integer("epoch", 2, "number of training epochs")
tf.app.flags.DEFINE_float("critic_lr", 1e-3, "critic learning rate")
tf.app.flags.DEFINE_float('gen_lr', 1e-3, "Generator learning rate")
tf.app.flags.DEFINE_integer('im_size', 32, "input image size")
tf.app.flags.DEFINE_integer('z_dim', 256, "input noise dimensionality")
config=tf.app.flags.FLAGS
config(sys.argv, known_only=True)

class Test:
    def __init__(self, d_path, s_path):
        self.d_path = d_path
        self.s_path = s_path
    def classifier(self, x, c, )