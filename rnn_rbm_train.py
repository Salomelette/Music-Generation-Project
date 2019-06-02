import tensorflow as tf
import numpy as np
import rnn_rbm
import extract_polyphonic
import os
import time
batch_size=100

def make_train(datapath,nb_epochs):
    saved_weights_path = "ckpt_dir/initialized.ckpt"
    x, generate, cost, W, bh, bv, learning_rate, Wuh, Wuv, Wvu, Wuu, bu, u0 = rnn_rbm.rnnrbm()
    variables = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    gradients = optimizer.compute_gradients(cost,variables)
    gradients = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gradients] #We use gradient clipping to prevent gradients from blowing up during training
    update = optimizer.apply_gradients(gradients)

    database = os.listdir(datapath)
    musics = extract_polyphonic.extract_poly2(database)

    ckpt = tf.train.Saver(variables,max_to_keep=1000)

    with tf.Session() as session:
        init = tf.initializers.global_variables()
        session.run(init)
        ckpt.restore(session, saved_weights_path)
        # saved_weights_path ? weight_initializations ? Peut-etre a faire pour opt 
        print("start")

        for epoch in range(nb_epochs):
            h_cost= []
            t=time.time()
            for music in musics:
                for i in range(0,len(music),batch_size):
                    datax =music[i:i+batch_size]
                    
                    eps=min(0.01,0.1/(i+1)) # decreasing learning rate according to another program (a trouver).
                    _,C = session.run([update,cost],feed_dict={x:datax,learning_rate:eps})

                    h_cost.append(C)
            print("epoch:{} cost:{} temps:{}".format(epoch,np.mean(h_cost),time.time()-t))
            print("C={}".format(C))

            if epoch % 5==0 and epoch!=0:
                ckpt.save(session,"ckpt_dir/epoch_bis_{}.ckpt".format(epoch))

    # mais je comprend pas on retourne rien ?? c'est passé ou tout ce qu'on a appris *legerement perdu*

if __name__=="__main__":
    make_train("database",101)