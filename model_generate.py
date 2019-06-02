from make_train import build_model
from generateur import create_midi_file
import pickle as pkl
import tensorflow as tf
import numpy as np
from collections import Counter
import scipy
import utils
def generate_music(model,notes2int,int2notes):

    num_generate = 1000

    input_eval = [notes2int["BOT"]]
    input_eval = tf.expand_dims(input_eval, 0)

    notes_generated = []

    temperature = 1.0
    model.reset_states()
    #while "EOT" not in notes_generated and len(notes_generated)<num_generate:
    for i in range(num_generate):
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
        if res[i]=="BOT" or res[i]=="EOT":
            continue
        velo_keys=list(vel[tuple(res[i])].keys())
        velo_values=list(vel[tuple(res[i])].values())
        tirage=np.random.choice(len(velo_keys), 1, p=velo_values)[0]
        if res[i][0]=="PAUSE":
            res2.append((64,res[i][1],0))
        else:
            res2.append((res[i][0],res[i][1],velo_keys[tirage]))
    print(i)
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
    #model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.load_weights(checkpoint_dir+"/ckpt_100")
    model.build(tf.TensorShape([1, None]))
    tab=[]

    with open('database.p','rb') as file:
        base,_,nb_occ=pkl.load(file)
    total=sum([item for item in nb_occ.values()])
    #nb_occ={k:i/total for k,i in nb_occ.items()}
    max_seq_lstm=[]
    tab_2=[]
    for i in range(10):
        res=generate_music(model,notes2int,int2notes)
        #print(res)
        if res[-1]=="EOT":
            res=res[1:-1]
        else:
            res=res[1:]
        all_notes=list(res)
        distrib=Counter(all_notes)
        #distrib={k:i/len(all_notes) for k,i in distrib.items()}
        index=dict()
        list_val=[]
        for i,k in enumerate(nb_occ.items()):
            index[k[0]]=i
            list_val.append(k[1]/total)
        #print(index)
        #print(list_val)
        vec_distrib=np.zeros(len(list_val))
        for k,v in distrib.items():
            if k not in index.keys():
                continue
            vec_distrib[index[k]]=v/len(all_notes)
        soft_vec=scipy.special.softmax(vec_distrib)

        soft_occ=scipy.special.softmax(list_val)
        #print(soft_occ)
        #distrib=[item for item in distrib.values()]
        #distrib=scipy.special.softmax(distrib)
        #somme=sum([d*np.log2(d) for d in distrib])
        #print(distrib)
        distrib=Counter(all_notes)
        distrib={k:i/len(all_notes) for k,i in distrib.items()}
        somme=sum([np.log(distrib.setdefault(i,1)) for i in nb_occ.keys() ])/len(nb_occ)
        print(somme)
        #max_seq_lstm.append(utils.check_sub_seq(base,res))
        #print(somme)
        print("Perplexité LSTM",2**(-somme))
        a=2**(-somme)
        a=np.exp(-somme)
        tab.append(a)
        #print(res)
        decode_and_create("LSTM.mid",res)


    print("moyenne Perplexité LSTM: ",np.mean(tab))
    print(max_seq_lstm)
    #print("moyenne plus long sous chaine commune cdm bpe: ",np.mean(max_seq_lstm))