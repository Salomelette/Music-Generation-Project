import mido
import numpy as np
import extract_polyphonic
import tensorflow as tf
import random
import rbm

nb_notes=127
seq_lengh=10
input_size=nb_notes*seq_lengh
n_hidden=50
n_hidden_RNN=256

def rnnrbm():

	v = tf.placeholder(tf.float64, [None,input_size])
	learning_rate = tf.placeholder(tf.float64)
	size_bt = tf.shape(x)[0]

	W   = tf.Variable(tf.zeros([input_size, n_hidden]), name="W")
    Wuh = tf.Variable(tf.zeros([n_hidden_RNN, n_hidden]), name="Wuh")
    Wuv = tf.Variable(tf.zeros([n_hidden_RNN, input_size]), name="Wuv")
    Wvu = tf.Variable(tf.zeros([input_size, n_hidden_RNN]), name="Wvu")
    Wuu = tf.Variable(tf.zeros([n_hidden_RNN, n_hidden_RNN]), name="Wuu")
    bh  = tf.Variable(tf.zeros([1, n_hidden]), name="bh")
    bv  = tf.Variable(tf.zeros([1, n_visible]), name="bv")
    bu  = tf.Variable(tf.zeros([1, n_hidden_RNN]), name="bu")
    u0  = tf.Variable(tf.zeros([1, n_hidden_RNN]), name="u0")
    BH_t = tf.Variable(tf.zeros([1, n_hidden]), name="BH_t") #?
	BV_t = tf.Variable(tf.zeros([1, input_size]), name="BV_t") #?




