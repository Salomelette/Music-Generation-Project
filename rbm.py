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


def get_cd_update(x, W, bv, bh, k, lr):
    #This is the contrastive divergence algorithm. 

    #First, we get the samples of x and h from the probability distribution
    #The sample of x
    x_sample = gibbs_sampling(x, W, bh, bv, k)
    #The sample of the hidden nodes, starting from the visible state of x
    h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
    #The sample of the hidden nodes, starting from the visible state of x_sample
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

    #Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
    lr = tf.constant(lr, tf.float32) #The CD learning rate
    size_bt = tf.cast(tf.shape(x)[0], tf.float32) #The batch size
    W_  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
    bv_ = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
    bh_ = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

    #When we do sess.run(updt), TensorFlow will run all 3 update steps
    updt = [W.assign_add(W_), bv.assign_add(bv_), bh.assign_add(bh_)]
    return updt
