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
n_hidden_RNN=100

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

    def generate_recurrence(count,k,u_tm1,primer,v_t,music): # k ?
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))

        #Run the Gibbs step to get the music output. Prime the RBM with the previous musical output.
        v_t_upt = rbm.gibbs_sampling(primer, W, bh_t, bv_t, k=25) #25 comme dans la source mais pas obligé si trop long 

        #Update the RNN hidden state based on the musical output and current hidden state.
        u_t = tf.tanh(bu + tf.matmul(v_t_upt, Wvu) + tf.matmul(u_tm1, Wuu))

        #Add the new output to the musical piece
        music = tf.concat([music,v_t_upt],0) #c'est bien dans ce sens la j'ai verifié 
        return count+1, k, u_t, v_t_upt, v_t, music 

    def generate(k,v=v,batch_size=batch_size,u0=u0,n_visible=input_size,prime_length=100):
        """this function handles generating music. This function is one of the outputs of the build_rnnrbm function
        Args:
            num (int): The number of timesteps to generate
            x (tf.placeholder): The data vector. We can use feed_dict to set this to the music primer. 
            size_bt (tf.float32): The batch size
            u0 (tf.Variable): The initial state of the RNN
            n_visible (int): The size of the data vectors
            prime_length (int): The number of timesteps into the primer song that we use befoe beginning to generate music
        Returns:
            The generated music, as a tf.Tensor"""
        U_first = tf.scan(rnn_step,v,initializer=u0)
        print("allo ",np.floor(prime_length/1))
        U = U_first[int(np.floor(prime_length/seq_lengh)), :, :] #je comprend pas ce que je fais (compréhension à partager si acquise)
        [_,_,_,_,_,music] = tf.while_loop(lambda count, num_iter, *args: count < num_iter, 
                                            generate_recurrence, [tf.constant(1,tf.int32), tf.constant(k), U, 
                                            tf.zeros([1, n_visible], tf.float32), v, tf.zeros([1, n_visible], tf.float32)],shape_invariants=[tf.constant(1,tf.int32).get_shape(), tf.constant(k).get_shape(), U.get_shape(), tf.TensorShape([1, n_visible]),v.get_shape(), tf.TensorShape([None, None])]) #ca non plus

        return music 



    tf.assign(BH_t,tf.tile(BH_t,[batch_size,1]))
    tf.assign(BV_t,tf.tile(BV_t,[batch_size,1]))

    u_t = tf.scan(rnn_step,v,initializer=u0)

    BV_t = tf.reshape(tf.scan(rnn_bias_visible,u_t,tf.zeros([1,input_size],tf.float32)),[batch_size,input_size])
    BH_t = tf.reshape(tf.scan(rnn_bias_hidden,u_t,tf.zeros([1,n_hidden],tf.float32)),[batch_size,n_hidden])

    cost = rbm.free_energy_cost(v,W,BH_t,BV_t,k=15)

    return v, generate, cost, W, bh, bv, learning_rate, Wuh, Wuv, Wvu, Wuu, bu, u0

    