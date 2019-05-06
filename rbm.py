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
		hk = sample(tf.sigmoid(tf.matmul(W.T,vt) + bh))
		vt = sample(tf.sigmoid(tf.matmul(W,hk) + bv))

	#stop_gradient ? 

	return vt 



