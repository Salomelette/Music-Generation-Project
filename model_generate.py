from make_train import build_model
from generateur import create_midi_file
import pickle as pkl
import tensorflow as tf
import numpy as np

def generate_music(model,notes2int,int2notes):

    num_generate = 3000

    input_eval = [notes2int["BOT"]]
    input_eval = tf.expand_dims(input_eval, 0)

    notes_generated = []

    temperature = 1.0
    model.reset_states()
    while "EOT" not in notes_generated and len(notes_generated)<num_generate:
    #for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        #print(predictions.shape)
        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        #predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        notes_generated.append(int2notes[predicted_id])

    return ["BOT"]+notes_generated

def decode_and_create(filename,res):
    
    res2=[]
    for i in range(len(res)):
        velo_keys=list(vel[tuple(res[i])].keys())
        velo_values=list(vel[tuple(res[i])].values())
        tirage=np.random.choice(len(velo_keys), 1, p=velo_values)[0]
        if res[i][0]=="PAUSE":
            res2.append((64,res[i][1],velo_keys[tirage]))
        else:
            res2.append((res[i][0],res[i][1],velo_keys[tirage]))
    # print(len(res2))
    #print(res2)
    create_midi_file(res2,filename)

if __name__=="__main__":
    with open("model_data.p",'rb') as file:
        data=pkl.load(file)
    nb_occ=data["occ"]
    notes2int=data["n2i"]
    int2notes=data["i2n"]
    vel=data["vel"]
    vocab_size=len(nb_occ)
    
    checkpoint_dir = './training_checkpoints'
    model = build_model(vocab_size, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    
    res=generate_music(model,notes2int,int2notes)
    if res[-1]=="EOT":
        res=res[1:-1]
    else:
        res=res[1:]
    #print(res)
    decode_and_create("premier_test9.mid",res)
    