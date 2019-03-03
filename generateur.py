import json
import numpy as np
import pickle as pkl
from mido import Message, MidiFile, MidiTrack

def create_midi_file(notes,filename):
    track=MidiTrack()
    outfile=MidiFile()
    outfile.tracks.append(track)
    track.append(Message('program_change', program=12))

    for note in notes:
        track.append(Message('note_on', note=note[0], velocity=note[2], time=note[1]))
        track.append(Message('note_off', note=note[0], velocity=note[2], time=note[1]))

    outfile.save(filename)

def generate_music(filename):
    with open('markov_model.p','rb') as file:
        data=pkl.load(file)
    A=data['A']
    pi=data['pi']
    vel=data['velocity']
    nb_notes=data['nb_notes']
    #nb_notes=20
    index=data['index']
    index_inv=dict()
    for key in index.keys():
        index_inv[index[key]]=key
    dim=len(A)
    res=[np.random.choice(dim, 1, p=pi)[0]]
    print(res)
    for j in range(nb_notes-1):
        res.append(np.random.choice(dim, 1, p=A[res[-1]])[0])
    print(res)
    res2=[]
    for i in range(len(res)):
        velo_keys=list(vel[index_inv[res[i]]].keys())
        velo_values=list(vel[index_inv[res[i]]].values())
        tirage=np.random.choice(len(velo_keys), 1, p=velo_values)[0]

        res2.append((index_inv[res[i]][0],index_inv[res[i]][1],velo_keys[tirage]))
    print(res2)
    create_midi_file(res2,filename)


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
    table_replacement=data['table_replacement']
    #nb_notes=20
    index=data['index']
    index_inv=dict()
    for key in index.keys():
        index_inv[index[key]]=key
    dim=len(A)

    res=[np.random.choice(dim, 1, p=pi)[0]]
    for j in range(nb_notes-1):
        res.append(np.random.choice(dim, 1, p=A[res[-1]])[0])
    res2=[]
    print(len(res))
    for i in range(len(res)):
        val=index_inv[res[i]]
        if type(val) is str:
            decode=decode_bigram(val,table_replacement)
            
        else:
            decode=[index_inv[res[i]]]
        print(len(decode))
        for item in decode:
            velo_keys=list(vel[item].keys())
            velo_values=list(vel[item].values())
            tirage=np.random.choice(len(velo_keys), 1, p=velo_values)[0]

            res2.append((item[0],item[1],velo_keys[tirage]))

    create_midi_file(res2,filename)
# def generate_music_old():
#     with open('markov_model.json','r') as file:
#         data=file.read()
#     data=json.loads(data)
#     A=data['A']
#     pi=data['pi']
#     vel=data['velocity']
#     temps=data['temps']
    
#     #nb_notes=10
#     doublets=data['doublets']
#     set_notes=list(range(12))+doublets
#     is_doublet=False
#     cs=np.cumsum(pi)
#     a=np.random.rand(1)
#     i=0
#     while cs[i]<a:
#         i+=1
#     old_val=i
#     res=[old_val]
#     if old_val>=12:
#         is_doublet=True
#     for j in range(nb_notes-1):
#         if is_doublet:
#             is_doublet=False
#             continue
#         cs=np.cumsum(A[old_val])
#         a=np.random.rand(1)
#         i=0
#         while cs[i]<a:
#             i+=1
#         old_val=i
#         if old_val>=12:
#             is_doublet=True
#         res.append(old_val)
#     with_doublet=[]
#     for item in res:
#         if item<12:
#             with_doublet.append(item)
#         else:
#             with_doublet.append(set_notes[item][0])
#             with_doublet.append(set_notes[item][1])
#     create_midi_file(with_doublet,vel,temps)


# for i in range(10):
#     generate_music("test_{}.mid".format(i))
generate_music_order('first_test.mid')