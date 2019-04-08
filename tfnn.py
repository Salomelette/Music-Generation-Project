#https://www.tensorflow.org/tutorials/sequences/text_generation
from extraction_fichiers import extract
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
#from discord_hooks import Webhook
from sklearn.model_selection import GridSearchCV
from generateur import create_midi_file

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
    if len(track)!=0:
        notes_dataset = tf.data.Dataset.from_tensor_slices(track)
        dataset_Final=dataset_Final.concatenate(notes_dataset)
    
dataset=dataset_Final

 
sequences = dataset.batch(sequence_length+1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


network_input = []
network_output = []
for track in notes:
    for i in range(0, len((track)) - sequence_length, 1):
        sequence_in = track[i:i + sequence_length]
        sequence_out = track[i+1:i + sequence_length+1]
        network_input.append(sequence_in)
        network_output.append(sequence_out)

training_data=[(network_input[i],network_output[i]) for i in range(len(network_input))]

DATASET_SIZE=len(training_data)
train_size = int(0.8 * DATASET_SIZE)
test_size = int(0.20 * DATASET_SIZE)

examples_per_epoch = sum([len(track) for track in notes])//sequence_length
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

data_train=dataset.take(train_size)
data_test=dataset.skip(train_size)


# Length of the vocabulary in chars
vocab_size = len(nb_occ.keys())

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNLSTM
  print("oui")
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.LSTM, recurrent_activation='sigmoid')
  
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]))
    model.add(rnn(rnn_units,return_sequences=True,recurrent_initializer='glorot_uniform',stateful=True))
    model.add(tf.keras.layers.Dense(vocab_size))

    return model  
  

        
model = build_model(vocab_size = vocab_size,embedding_dim=embedding_dim,rnn_units=rnn_units,batch_size=BATCH_SIZE)


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#example_batch_loss  = loss(target_example_batch, example_batch_predictions)
#print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)") 
#print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer = tf.train.AdamOptimizer(),loss = loss)
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

Early_Stopping=tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.01,patience=5)

EPOCHS=100
history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback,Early_Stopping])

#rebuild du model pour accepter un nouevau batch_size
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

def generate_music(model, start_notes):

  num_generate = 1000

  input_eval = [notes2int[start_notes]]
  input_eval = tf.expand_dims(input_eval, 0)

  notes_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      
      notes_generated.append(int2notes[predicted_id])

  return [start_notes]+notes_generated


#clf = GridSearchCV(sk_nn,param_grid,cv=4,n_jobs=1,verbose=2)
#
#clf.fit(X,y)
#ostr =[]
#ostr.append("Best parameters set found on development set:")
##ostr.append("")
#ostr.append(str(clf.best_params_))
#ostr.append("")
#ostr.append("Grid scores on development set:")
##ostr.append("")
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    ostr.append("%0.3f (+/-%0.03f) for %r"
#          % (mean, std * 2, params))
##   ostr.append("")
#
#ostr = '\n'.join(ostr)
#print(ostr)
#with open(outfile+'.txt','w') as f:
#    f.write(ostr)
#pickle.dump(clf.cv_results_,open(outfile+'.p','wb'))






res=generate_music(model,list(nb_occ.keys())[0])
res2=[]
for i in range(len(res)):
    #↨print(i)
    velo_keys=list(vel[tuple(res[i])].keys())
    velo_values=list(vel[tuple(res[i])].values())
    tirage=np.random.choice(len(velo_keys), 1, p=velo_values)[0]

    res2.append((res[i][0],res[i][1],velo_keys[tirage]))
filename="allo_seq_len5_beth.mid"
create_midi_file(res2,filename)
