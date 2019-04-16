import numpy as np
import tensorflow as tf
from extraction_fichiers import extract,extract_train_test
import os
import pickle as pkl

def prepareData(path,sequence_length=10):

    database = os.listdir(path)
    notes, vel, nb_occ= extract(database)

    for track in notes:
        track.insert(0,"BOT")
        track.append("EOT")
        nb_occ["BOT"]+=1
        nb_occ["EOT"]+=1

    notes2int={u:i for i,u in enumerate(nb_occ.keys())}
    int2notes = np.array(list(nb_occ.keys()))
    notes_as_int = [np.array([notes2int[c] for c in track]) for track in notes]

    def generator():
        for track in notes_as_int:
            for i in range(0, len((track)) - sequence_length, 1):
                sequence_in = track[i:i + sequence_length]
                sequence_out = track[i+1:i + sequence_length+1]
                yield sequence_in,sequence_out
#    gen=generator()
#    for i in gen:
#        print(i)
    dataset = tf.data.Dataset.from_generator(generator,(tf.int64,tf.int64))

    with open("model_data.p",'wb') as file:
        pkl.dump({"occ":nb_occ,"n2i":notes2int,"i2n":int2notes,"vel":vel},file)
    return dataset,nb_occ,notes