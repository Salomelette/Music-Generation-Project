
#https://stackoverflow.com/questions/43321950/synthesia-plays-well-midi-file-without-any-note-off-event

import numpy as np
import mido 
import os 
import time
import json
import pickle as pkl
from collections import Counter
from download_midi_file import get_mid_file

try:
    database = os.listdir("./database")
except FileNotFoundError as e:
    print("Téléchargement de la base de donnée")
    get_mid_file()
    
def convert_midibd(database):
    bd = []
    for m in database:
        try:
            bd.append(mido.MidiFile("./database/"+str(m)))
        except IOError:
            print('{} MThd not found. Probably not a MIDI file'.format(m))
    return bd 


#def extract(database):
#    try :
#        with open('database.p','rb') as file:
#            data=pkl.load(file)
#            print("fichier trouvé")
#            return data[0],data[1],data[2]
#    except Exception as e:
#        print(e)
#        print("lecture des fichiers")
#    res = []
#    midibd = convert_midibd(database)
#    velocity = dict()
#    total = []
#    for m in midibd:
#        resm = []
#        for track in m.tracks:
#            for msg in track:
#                print(msg)
#                if not msg.is_meta and msg.type == 'note_on':
#                    
#                    if msg.time != 0 and msg.velocity!=0:
#                        note = msg.note,msg.time
#                        if msg.time>5:
#                            note = msg.note,round(msg.time,-1)
#                        else:
#                            note=msg.note,1
#                        resm.append(note)
#                        total.append(note)
#                        vel = msg.velocity
#                        if note not in velocity.keys():
#                            velocity[note]=Counter()
#                        velocity[note][vel] += 1
#
#        res.append(resm)
#
#    nb_occ = Counter(total)
#    for i in nb_occ:
#        for keys in velocity[i]:
#            velocity[i][keys] /= nb_occ[i]
#
#    with open('database.p','wb') as file:
#        pass
#        pkl.dump([res,velocity,nb_occ],file)
#    return res, velocity, nb_occ

def extract(database):
    try :
        with open('database.p','rb') as file:
            data=pkl.load(file)
            print("fichier trouvé")
            return data[0],data[1],data[2]
    except Exception as e:
        print(e)
        print("lecture des fichiers")
    res = []
    midibd = convert_midibd(database)
    velocity = dict()
    total = []
    note_on=False
    for m in midibd:
        resm = []
        for track in m.tracks:
            for msg in track:
                #print(msg)
                if not msg.is_meta and (msg.type == 'note_on' or msg.type=='note_off'):
                    if not note_on and msg.velocity!=0:
                        stock_velocity=msg.velocity
                        note_on=True
                    else:
                        note = msg.note,msg.time
                        if msg.time>5:
                            note = msg.note,round(msg.time,-1)
                        else:
                            note=msg.note,1
                        resm.append(note)
                        total.append(note)
                        vel = stock_velocity
                        if note not in velocity.keys():
                            velocity[note]=Counter()
                        velocity[note][vel] += 1
                        note_on=False

        res.append(resm)

    nb_occ = Counter(total)
    for i in nb_occ:
        for keys in velocity[i]:
            velocity[i][keys] /= nb_occ[i]

    with open('database.p','wb') as file:
        pkl.dump([res,velocity,nb_occ],file)
    return res, velocity, nb_occ

def extract_train_test(database,test_size):
    try :
        with open('database_test.p','rb') as file:
            data=pkl.load(file)
            print("fichier trouvé")
            return data[0],data[1],data[2],data[3]
    except Exception as e:
        print(e)
        print("lecture des fichiers")


    res = []
    midibd = convert_midibd(database)
    velocity = dict()
    total = []
    note_on=False
    for m in midibd[:int(len(midibd)*(1-test_size))]:
        resm = []
        for track in m.tracks:
            for msg in track:
                if not msg.is_meta and (msg.type == 'note_on' or msg.type=='note_off'):
                    if not note_on and msg.velocity!=0:
                        stock_velocity=msg.velocity
                        note_on=True
                    else:
                        note = msg.note,msg.time
                        if msg.time>5:
                            note = msg.note,round(msg.time,-1)
                        else:
                            note=msg.note,1
                        resm.append(note)
                        total.append(note)
                        vel = stock_velocity
                        if note not in velocity.keys():
                            velocity[note]=Counter()
                        velocity[note][vel] += 1
                        note_on=False

        res.append(resm)
    test=[]
    note_on=False
    for m in midibd[int(len(midibd)*(1-test_size)):]:
        resm = []
        for track in m.tracks:
            for msg in track:
                if not msg.is_meta and (msg.type == 'note_on' or msg.type=='note_off'):
                    if not note_on and msg.velocity!=0:
                        stock_velocity=msg.velocity
                        note_on=True
                    else:
                        note = msg.note,msg.time
                        if msg.time>5:
                            note = msg.note,round(msg.time,-1)
                        else:
                            note=msg.note,1
                        resm.append(note)
                        total.append(note)

        test.append(resm)

    nb_occ = Counter(total)
    for i in nb_occ:
        if i in velocity.keys():
            for keys in velocity[i]:
                velocity[i][keys] /= nb_occ[i]
                

    with open('database_test.p','wb') as file:
        pkl.dump([res,velocity,nb_occ,test],file)
    return res, velocity, nb_occ,test

def find_bigram(list_track,nb):
    replacement_table=dict()
    for k in range(nb):
        count=Counter()
        for track in list_track:
            for i in range(len(track)-1):
                count[(track[i],track[i+1])]+=1
        sort=sorted(count.items(),key=lambda x:x[1],reverse=True)
        if sort[0][1]==1:
            break
        most_occ=sort[0][0]
        replacement_table["bigram_{}".format(k)]=most_occ
        for track in list_track:
            L=len(track)-1
            i=0
            while i <L:
                if track[i]==most_occ[0] and track[i+1]==most_occ[1]:
                    track.insert(i,"bigram_{}".format(k))
                    a=track.pop(i+1)
                    b=track.pop(i+1)
                L=len(track)-1
                i+=1

    with open("replacement_table.txt",'w') as file:
        file.write(str(replacement_table))
    return list_track,replacement_table


def learn_markov_model_order(list_track,vel):
    new_list_track,table_replacement=find_bigram(list_track,round(0.3*len(vel.keys())))
    notes =list(vel.keys())+list(table_replacement.keys())
    dim=len(notes)
    pi = np.ones(dim)
    A = np.ones(dim*dim).reshape(dim,dim)

    index = dict()
    for i in range(len(notes)):
        index[notes[i]] = i 
    
    for track in list_track:
        if len(track)==0:
            continue
        pi[index[track[0]]]+=1

        for i in range(len(track)-1):
            A[index[track[i]]][index[track[i+1]]]+=1

    n=np.linalg.norm(pi,ord=1)
    pi/=n
    for item in A:
        n=np.linalg.norm(item,ord=1)
        item/=n

    nb_notes=int(np.mean(np.array([len(track) for track in list_track if len(track)!=0])))
    res=dict()
    res['A']=A
    res['pi']=pi
    res['velocity']=vel
    res['nb_notes']=nb_notes
    res['index']=index
    res['table_replacement']=table_replacement

    with open('markov_model_order.p','wb') as file:
        pkl.dump(res,file)


def learn_markov_model(list_track,vel):
    dim = len(vel)
    pi = np.ones(dim)
    A = np.ones(dim*dim).reshape(dim,dim)
    index = dict()
    notes = list(vel.keys())

    for i in range(len(notes)):
        index[notes[i]] = i 

    for track in list_track:
        if len(track)==0:
            continue
        pi[index[track[0]]]+=1
        for i in range(len(track)-1):
            A[index[track[i]]][index[track[i+1]]]+=1
    n=np.linalg.norm(pi,ord=1)
    pi/=n
    for item in A:
        n=np.linalg.norm(item,ord=1)
        item/=n
    nb_notes=int(np.mean(np.array([len(track) for track in list_track if len(track)!=0])))
    res=dict()
    res['A']=A
    res['pi']=pi
    res['velocity']=vel
    res['nb_notes']=nb_notes
    res['index']=index
    with open('markov_model.p','wb') as file:
        pkl.dump(res,file)



if __name__=="__main__":
    notes, vel, nb_occ = extract(database)
    with open("notes.txt",'w') as file:
        file.write(str(nb_occ))
    with open("velocity_distribution.txt",'w') as file:
        file.write(str(vel))
    with open("notes.txt",'w') as file:
        file.write(str(notes))
    #learn_markov_model(notes,vel)
    t=time.time()
    learn_markov_model_order(notes,vel)
    print(time.time()-t)
    
