import tensorflow as tf
import numpy as np
import rnn_rbm
import extract_polyphonic
import os


def train(datapath,nb_epochs):

    x,cost, W,bh,bv, learning_rate, Wuh,Wuv,Wvu,Wuu,bu,u0 = rnn_rbm.rnnrbm()
    variables = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    optimizer = tf.train,AdamOptimizer(learning_rate=learning_rate)
    gradients = optimizer.compute_gradient(cost,variables)
    gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients] #We use gradient clipping to prevent gradients from blowing up during training
    update = optimizer.apply_gradients(gradients)

    database = os.listdir(datapath)
    musics = extract_polyphonic.extract_poly(database)

    ckpt = tf.train.Saver(variables)