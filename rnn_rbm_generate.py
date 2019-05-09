import tensorflow as tf
import numpy as np
import rnn_rbm
import extract_polyphonic  

database = os.listdir("./database")

#num correspond aux nombres de chansons a generer 
def generate_music(num,ckpt_datapath):

    x, generate, cost, W, bh, bv, learning_rate, Wuh, Wuv, Wvu, Wuu, bu, u0 = rnn_rbm.rnnrbm()
    variables = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    saver = tf.train.Saver(variables) #ha c'est là qu'il a tout gardé ??

    songs = extract_polyphonic.extract_poly(database) 

    with tf.Session() as session:
    	init = tf.initializers.global_variables()
    	session.run(init)

    	#saver.restore(session, ckpt_datapath) a faire ou pas ?? 

    	for i in tqdm(range(num)):
    		generated_music = session.run(generate(300), feed_dict={x: songs})
    		new_music_filename = "generated_music_{}.midi".format(i)
    		extract_polyphonic.write_midi_poly(generated_music,new_music_filename)








