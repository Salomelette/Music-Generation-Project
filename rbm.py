import mido
import numpy as np
import extract_polyphonic
import tensorflow as tf
import random

def sample(probs):
    #Renvoie un vecteur de 0 et de 1 sample a partir de celui qui est rentrée 
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

"""def gibbs_sampling(vt,W,bh,bv,k):

    for i in range(k):
        hk = sample(tf.sigmoid(tf.matmul(vt,W) + bh))
        vt = sample(tf.sigmoid(tf.matmul(hk,tf.transpose(W)) + bv))

    #stop_gradient ? 

    return vt"""

def gibbs_sampling(x, W, bh, bv, k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
        return count+1, k, xk

    #Run gibbs steps for k iterations
    ct = tf.constant(0) #counter
    [_, _, x_sample] = tf.while_loop(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x])
    #We need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

def free_energy_cost(vt,W,bh,bv,k):

    v_sample = gibbs_sampling(vt,W,bh,bv,k)

    def F(v):
        return  -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(v, W) + bh)), 1) - tf.matmul(v, tf.transpose(bv))
    
    cost = tf.reduce_mean(tf.subtract(F(vt),F(v_sample)))

    return cost