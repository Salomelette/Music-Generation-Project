import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
from datatools import prepareData,prepareData_test
import pickle as pkl
import pdb

  
def build_model(vocab_size, embedding_dim=256, rnn_units=1024, batch_size=64):
    if tf.test.is_gpu_available():
        rnn = tf.keras.layers.CuDNNLSTM
    else:
        import functools
        rnn = functools.partial(
        tf.keras.layers.LSTM, recurrent_activation='sigmoid')


    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]))
    model.add(rnn(rnn_units,return_sequences=True,stateful=True))
    model.add(rnn(rnn_units,return_sequences=True))
    model.add(rnn(rnn_units,return_sequences=True))
    model.add(tf.keras.layers.Dense(vocab_size))
    #model.add(tf.keras.layers.Activation('softmax'))
    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def train_model(datapath,sequence_length,batch_size,nb_epoch=100):
    dataset,nb_occ,notes=prepareData(datapath,sequence_length)
    vocab_size=len(nb_occ)
    examples_per_epoch = sum([len(track) for track in notes])//sequence_length
    steps_per_epoch = examples_per_epoch//batch_size
    
    buffer_size=10000
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    model = build_model(vocab_size = vocab_size,batch_size=batch_size)
    model.compile(optimizer = tf.train.AdamOptimizer(),loss = loss)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True,period=5)
    
    history = model.fit(dataset.repeat(), epochs=nb_epoch, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

def train_model_test(datapath,sequence_length,batch_size,nb_epoch=100):
    dataset,nb_occ,notes,dataset_test,notes_test=prepareData_test(datapath,sequence_length,test=0.2)
    vocab_size=len(nb_occ)
    examples_per_epoch = sum([len(track) for track in notes])//sequence_length
    steps_per_epoch = examples_per_epoch//batch_size
    buffer_size=10000
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    model = build_model(vocab_size = vocab_size,batch_size=batch_size)
    model.compile(optimizer = tf.train.AdamOptimizer(),loss = loss)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True,period=5)
    Early_Stopping=tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.001,patience=10)
    
    history = model.fit(dataset.repeat(), epochs=nb_epoch, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
    
    
    examples_per_epoch = sum([len(track) for track in notes_test])//sequence_length
    steps_per_epoch = examples_per_epoch//batch_size
    dataset_test = dataset_test.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    score_test=model.evaluate(dataset_test,steps=steps_per_epoch)
    print(score_test)
if __name__=="__main__":
    #train_model(datapath="./database",sequence_length=10,batch_size=4,nb_epoch=100)
    
    train_model_test(datapath="./database",sequence_length=100,batch_size=64,nb_epoch=100)