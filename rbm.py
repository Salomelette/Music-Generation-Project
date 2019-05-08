import mido
import numpy as np
import extract_polyphonic
import tensorflow as tf
import random

def sample(probs):
    #Renvoie un vecteur de 0 et de 1 sample a partir de celui qui est rentrée 
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def gibbs_sampling(vt,W,bh,bv,k):

    for i in range(k):
        hk = sample(tf.sigmoid(tf.matmul(vt,W) + bh))
        vt = sample(tf.sigmoid(tf.matmul(hk,tf.transpose(W)) + bv))

    #stop_gradient ? 

    return vt

def free_energy_cost(vt,W,bh,bv,k):

    v_sample = gibbs_sampling(vt,W,bh,bv,k)

    def F(v):
        return  -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(v, W) + bh)), 1) - tf.matmul(v, tf.transpose(bv))
    
    cost = tf.reduce_mean(tf.subtract(F(vt),F(v_sample)))

    return cost
