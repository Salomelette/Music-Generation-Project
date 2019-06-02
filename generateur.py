import json
import numpy as np
import pickle as pkl
from mido import Message, MidiFile, MidiTrack
from collections import Counter
import utils

def create_midi_file(notes,filename):
    track=MidiTrack()
    outfile=MidiFile()
    outfile.tracks.append(track)
    track.append(Message('program_change', program=1))

    for note in notes:
        track.append(Message('note_on', note=note[0], velocity=note[2], time=0))
        track.append(Message('note_off', note=note[0], velocity=0, time=note[1]))

    outfile.save(filename)

def generate_music(filename):
    with open('markov_model.p','rb') as file:
        data=pkl.load(file)
    A=data['A']
    pi=data['pi']
    vel=data['velocity']
    nb_notes=data['nb_notes']
    index=data['index']
    index_inv=dict()
    all_notes=[]
    for key in index.keys():
        index_inv[index[key]]=key
    dim=len(A)
    res=[np.random.choice(dim, 1, p=pi)[0]]
    for j in range(int(nb_notes)-1):
        res.append(np.random.choice(dim, 1, p=A[res[-1]])[0])
    res2=[]
    for i in range(len(res)):
        all_notes.append(index_inv[res[i]])
        velo_keys=list(vel[index_inv[res[i]]].keys())
        velo_values=list(vel[index_inv[res[i]]].values())
        tirage=np.random.choice(len(velo_keys), 1, p=velo_values)[0]

        res2.append((index_inv[res[i]][0],index_inv[res[i]][1],velo_keys[tirage]))
    create_midi_file(res2,filename)
    distrib=Counter(all_notes)
    distrib={k:i/len(all_notes) for k,i in distrib.items()}
    #distrib=[item/len(all_notes) for item in distrib.values()]
    #somme=sum([d*np.log2(d) for d in distrib])
    #print(somme)
    #print("Perplexité CDM1",2**(-somme))
    return distrib,all_notes

def decode_bigram(key,table_replacement):
    bigram=table_replacement[key]
    res=[]
    for item in bigram:
        if type(item) is str:
            res+=decode_bigram(item,table_replacement)
        else:
            res.append(item)

    return res

def generate_music_order(filename):
    with open('markov_model_order.p','rb') as file:
        data=pkl.load(file)

    A=data['A']
    pi=data['pi']
    vel=data['velocity']
    nb_notes=data['nb_notes']
    #print(nb_notes)
    table_replacement=data['table_replacement']
    index=data['index']
    index_inv=dict()
    all_notes=[]
    for key in index.keys():
        index_inv[index[key]]=key
    dim=len(A)

    res=[np.random.choice(dim, 1, p=pi)[0]]
    for j in range(int(nb_notes)-1):
        res.append(np.random.choice(dim, 1, p=A[res[-1]])[0])
    res2=[]
    taille_decode=[]
    for i in range(len(res)):
        val=index_inv[res[i]]
        if type(val) is str:
            decode=decode_bigram(val,table_replacement)
            
        else:
            decode=[index_inv[res[i]]]
        taille_decode.append(len(decode))
        for item in decode:
            all_notes.append(item)
            velo_keys=list(vel[item].keys())
            velo_values=list(vel[item].values())
            tirage=np.random.choice(len(velo_keys), 1, p=velo_values)[0]

            res2.append((item[0],item[1],velo_keys[tirage]))
    with open("taille_decode.txt",'w') as file:
        file.write(str(taille_decode))
    create_midi_file(res2,filename)
    distrib=Counter(all_notes)
    distrib={k:i/len(all_notes) for k,i in distrib.items()}
    #distrib=[item/len(all_notes) for item in distrib.values()]
    #somme=sum([d*np.log2(d) for d in distrib])
    #print(somme)
    #print("Perplexité bpe",2**(-somme))
    return distrib,all_notes

if __name__=="__main__":
    p_1=[]
    p_bpe=[]
    max_seq1=[]
    max_seq_bpe=[]
    with open('database.p','rb') as file:
        res,_,nb_occ=pkl.load(file)
        #pkl.dump([res,velocity,nb_occ,test],file)
    total=sum([item for item in nb_occ.values()])
    nb_occ={k:i/total for k,i in nb_occ.items()}
    print(len(res))
    for i in range(200):

        a,seq1=generate_music('test_cdm1.mid')
        somme=sum([np.log2(a.setdefault(key,1)) for key in nb_occ.keys()])/len(nb_occ)
        a=2**-somme
        #print(seq1)
        #max_seq1.append(utils.check_sub_seq(res,seq1))
        b,seq2=generate_music_order('test_bpe.mid')
        #max_seq_bpe.append(utils.check_sub_seq(res,seq2))
        somme=sum([np.log2(b.setdefault(key,1)) for key in nb_occ.keys()])/len(nb_occ)
        #print(somme)
        b=2**-somme
        p_1.append(a)
        p_bpe.append(b)
    print("moyenne Perplexité cdm1: ",np.mean(p_1))
    print("moyenne Perplexité cdm bpe: ",np.mean(p_bpe))
    """
    print(max_seq1)
    print("moyenne plus long sous chaine commune cdm1: ",np.mean(max_seq1))
    print(max_seq_bpe)
    print("moyenne plus long sous chaine commune cdm bpe: ",np.mean(max_seq_bpe))"""