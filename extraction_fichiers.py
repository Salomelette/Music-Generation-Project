import numpy as np
import mido 
import os 
import time
import json
database = os.listdir("./database")

def convert_midibd(database):
    bd = []
    for m in database:
        try:
            bd.append(mido.MidiFile("./database/"+str(m)))
        except IOError:
            print('{} MThd not found. Probably not a MIDI file'.format(m))

    return bd 


def extract(database):
    res = []
    midibd = convert_midibd(database)
    velocity = dict()
    temps = dict()
    nb_occ = dict()
    
    for i in range(12):
        velocity[i] = 0
        temps[i] = 0
        nb_occ[i] = 0
        
        
    for m in midibd:
        resm = []
        for track in m.tracks:
            for msg in track:
                if not msg.is_meta and msg.type == 'note_on':
                    note = msg.note%12 
                    resm.append(note)
                    velocity[note] += msg.velocity 
                    temps[note] += msg.time
                    nb_occ[note] += 1 

        res.append(resm)
    
    for i in range(12):
        velocity[i] /= nb_occ[i]
        temps[i] /= nb_occ[i]
                   
    return res, velocity, temps, nb_occ

def find_doublet(list_track,nb):
    count=dict()
    for i in range(12):
        for j in range(12):
            count[(i,j)]=0
    for track in list_track:
        for i in range(len(track)-1):
            count[(track[i],track[i+1])]+=1
    res=sorted(count.items(), key=lambda x: x[1],reverse=True)
    return [res[i][0] for i in range(nb)]


def learn_markov(list_track,vel,temps):
    doublets = find_doublet(list_track,20)
    dim = 12 + len(doublets)
    pi = np.zeros(dim)
    A = np.zeros(dim*dim).reshape(dim,dim)

    for track in list_track:
        is_doublet = False
        if (track[0], track[1]) in doublets:
            pi[12+doublets.index((track[0], track[1]))] += 1
            if (track[2],track[3]) in doublets:
                A[12+doublets.index((track[0],track[1]))][12+doublets.index((track[2],track[3]))]+=1
            else:
                A[12+doublets.index((track[0],track[1]))][track[2]]+=1
            is_doublet = True
        else:
            pi[track[0]]+=1
        for i in range(len(track)-1):
            if is_doublet==True:
                is_doublet=False
                continue
            if (track[i], track[i+1]) in doublets:
                if i+2<len(track) and i+3<len(track) and (track[i+2],track[i+3]) in doublets:
                    A[12+doublets.index((track[i], track[i+1]))][12+doublets.index((track[i+2],track[i+3]))]+=1
                else:
                    if i+2<len(track):
                        A[12+doublets.index((track[i],track[i+1]))][track[i+2]]+=1
                    else:
                        continue
                is_doublet=True
            else:
                if i+2<len(track) and (track[i+1],track[i+2]) in doublets:
                    A[track[i]][12+doublets.index((track[i+1],track[i+2]))]+=1
                else:
                    A[track[i]][track[i+1]]+=1
    n=np.linalg.norm(pi,ord=1)
    pi/=n
    for item in A:
        n=np.linalg.norm(item,ord=1)
        item/=n
    print(A)
    print(sum(A[0]))
    print(pi)
    nb_notes=np.mean(np.array([len(track) for track in list_track]))
    res=dict()
    res['A']=A.tolist()
    res['pi']=pi.tolist()
    res['velocity']=vel
    res['temps']=temps
    res['nb_notes']=nb_notes
    with open('markov_model.json','w') as file:
        json.dump(res,file)
    
notes, vel, temps, nb_occ = extract(database)
#print(len(notes), vel, time, nb_occ)
print(len(notes))
learn_markov(notes,vel,temps)

    
