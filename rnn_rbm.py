import mido
import numpy as np
import extract_polyphonic
import tensorflow as tf
import random
import rbm

nb_notes=127
seq_lengh=10
input_size=nb_notes#*seq_lengh
n_hidden=50
n_hidden_RNN=256

def rnnrbm():

    v = tf.placeholder(tf.float32, [None,input_size])
    learning_rate = tf.placeholder(tf.float32)
    batch_size = tf.shape(v)[0]

    W   = tf.Variable(tf.zeros([input_size, n_hidden]), name="W")
    Wuh = tf.Variable(tf.zeros([n_hidden_RNN, n_hidden]), name="Wuh")
    Wuv = tf.Variable(tf.zeros([n_hidden_RNN, input_size]), name="Wuv")
    Wvu = tf.Variable(tf.zeros([input_size, n_hidden_RNN]), name="Wvu")
    Wuu = tf.Variable(tf.zeros([n_hidden_RNN, n_hidden_RNN]), name="Wuu")
    bh  = tf.Variable(tf.zeros([1, n_hidden]), name="bh")
    bv  = tf.Variable(tf.zeros([1, input_size]), name="bv")
    bu  = tf.Variable(tf.zeros([1, n_hidden_RNN]), name="bu")
    u0  = tf.Variable(tf.zeros([1, n_hidden_RNN]), name="u0")
    BH_t = tf.Variable(tf.zeros([1, n_hidden]), name="BH_t") #?
    BV_t = tf.Variable(tf.zeros([1, input_size]), name="BV_t") #?


    def rnn_step(u_m1,state):
        state = tf.reshape(state,[1,input_size])
        u = tf.tanh(tf.matmul(u_m1,Wuu)+tf.matmul(state,Wvu)+bu)
        return u

    def rnn_bias_visible(bv_t,u_m1):
        bv_t = tf.add(bv,tf.matmul(u_m1,Wuv))
        return bv_t

    def rnn_bias_hidden(bh_t,u_m1):
        bh_t = tf.add(bh,tf.matmul(u_m1,Wuh))
        return bh_t




    tf.assign(BH_t,tf.tile(BH_t,[batch_size,1]))
    tf.assign(BV_t,tf.tile(BV_t,[batch_size,1]))

    u_t = tf.scan(rnn_step,v,initializer=u0)

    BV_t = tf.reshape(tf.scan(rnn_bias_visible,u_t,tf.zeros([1,input_size],tf.float32)),[batch_size,input_size])
    BH_t = tf.reshape(tf.scan(rnn_bias_hidden,u_t,tf.zeros([1,n_hidden],tf.float32)),[batch_size,n_hidden])

    cost = rbm.free_energy_cost(v,W,BH_t,BV_t,k=10)

    return v,cost, W,bh,bv, learning_rate, Wuh,Wuv,Wvu,Wuu,bu,u0

    