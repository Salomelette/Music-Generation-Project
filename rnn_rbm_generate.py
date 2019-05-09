import tensorflow as tf
import numpy as np
import rnn_rbm


def generate_music(ckpt_datapath):


    x,cost, W,bh,bv, learning_rate, Wuh,Wuv,Wvu,Wuu,bu,u0 = rnn_rbm.rnnrbm()
    variables = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]
