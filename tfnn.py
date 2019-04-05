from extraction_fichiers import extract
import tensorflow as tf
#tf.enable_eager_execution()
import numpy as np
import os
import time
from discord_hooks import Webhook
url_web_hook='https://discordapp.com/api/webhooks/485826787574677564/jvaxi81nGdyoIxKng3LzqF9Fqh66tSPpolQ5vWSmVw7nYmfHAYiVfpaptmvZWveyitvG'
url_compte_jour='https://discordapp.com/api/webhooks/487302586072956928/cn99JoRVpyNLo8UvSvmuuFbsG7CzIdfGv5v8EMGcAhDcoMLbZNIjhho4SOVXtZpuUzCJ'
url_error='https://discordapp.com/api/webhooks/495517689545097246/B-_MUSOcJkAozN56jj-lFri6f84jvdLvbpY0K4hcdwcShzhrBPBaJHva3lVTHTsQN0J3'
#letsgetit

database = os.listdir("./database")
notes, vel, nb_occ = extract(database)



sequence_length = 10
notes2int={u:i for i,u in enumerate(nb_occ.keys())}
int2notes = np.array(list(nb_occ.keys()))
notes_as_int = [np.array([notes2int[c] for c in track]) for track in notes]

list_dataset=[]
dataset_Final=tf.data.Dataset.from_tensor_slices(notes_as_int[0])
for track in notes_as_int[1:]:
    notes_dataset = tf.data.Dataset.from_tensor_slices(track)
    dataset_Final=dataset_Final.concatenate(notes_dataset)
    
dataset=dataset_Final

#for i in dataset.take(1):
#  print(int2notes[i.numpy()])
  
sequences = dataset.batch(sequence_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(item)
    #print(repr(''.join(i[item.numpy()])))
    
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', input_example)
  print ('Target data:', target_example)
  
  
#for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
#    print("Step {}".format(i))
#    print("  input: {} ({})".format(input_idx, int_to_notes[int(input_idx)]))
#    print("  expected output: {} ({})".format(target_idx, int_to_notes[int(target_idx)]))
examples_per_epoch = sum([len(track) for track in notes])//sequence_length
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



# Length of the vocabulary in chars
vocab_size = len(nb_occ.keys())

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNGRU
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.GRU, recurrent_activation='sigmoid')
  
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                              batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model  
  

        
model = build_model(vocab_size = vocab_size,embedding_dim=embedding_dim,rnn_units=rnn_units,batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1): 
  example_batch_predictions = model(input_example_batch)
  
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()