import numpy as np
import mido 
import os 
import time
import json
import pickle as pkl
from collections import Counter

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
    total = []
    # temps = dict()
    # nb_occ = dict()
    
    # for i in range(12):
    #     velocity[i] = dict()
    #     temps[i] = dict()
    #     nb_occ[i] = 0
        
        
    for m in midibd:
        resm = []
        for track in m.tracks:
            for msg in track:
                if not msg.is_meta and msg.type == 'note_on':
                    if msg.time != 0 and msg.velocity!=0:
                        note = msg.note,msg.time
                        #note = msg.note,round(msg.time,-1)
                        resm.append(note)
                        total.append(note)
                        vel = msg.velocity
                        if note not in velocity.keys():
                            velocity[note]=Counter()
                        velocity[note][vel] += 1
                    # ti = msg.time
                    # if ti in temps[note]:
                    #     temps[note][ti] += 1
                    # else:
                    #     temps[note][ti] = 1
                    # nb_occ[note] += 1 

        res.append(resm)
    
    nb_occ = Counter(total)
    for i in nb_occ:
        for keys in velocity[i]:
            velocity[i][keys] /= nb_occ[i]
    #     for keys in temps[i]:
    #         temps[i][keys] /= nb_occ[i]

    return res, velocity, nb_occ

def find_bigram(list_track,nb):
    replacement_table=dict()

    for k in range(nb):
        count=Counter()
        for track in list_track:
            for i in range(len(track)-1):
                count[(track[i],track[i+1])]+=1
        sort=sorted(count.items(),key=lambda x:x[1],reverse=True)
        if sort[0][1]==1:
            print('c est break')
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
    print(replacement_table)
    return list_track,replacement_table

def find_doublet(list_track,nb): #bigramme
    count=dict()
    for i in range(12):
        for j in range(12):
            count[(i,j)]=0
    for track in list_track:
        for i in range(len(track)-1):
            count[(track[i],track[i+1])]+=1
    res=sorted(count.items(), key=lambda x: x[1],reverse=True)
    return [res[i][0] for i in range(nb)]

def learn_markov_model_order(list_track,vel):
    new_list_track,table_replacement=find_bigram(list_track,round(0.2*len(vel.keys())))
    print(len(table_replacement))
    notes =list(vel.keys())+list(table_replacement.keys())
    dim=len(notes)
    pi = np.ones(dim)
    A = np.ones(dim*dim).reshape(dim,dim)

    index = dict()
    for i in range(len(notes)):
        index[notes[i]] = i 
    
    for track in list_track:
        if len(track)==0:
            print('allo')
            continue
        pi[index[track[0]]]+=1

        for i in range(len(track)-1):
            A[index[track[i]]][index[track[i+1]]]+=1

    n=np.linalg.norm(pi,ord=1)
    pi/=n
    for item in A:
        n=np.linalg.norm(item,ord=1)
        #print( n)
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
    # doublets = find_doublet(list_track,10)
    dim = len(vel)
    pi = np.ones(dim)
    print(dim)
    A = np.ones(dim*dim).reshape(dim,dim)
    index = dict()
    notes = list(vel.keys())

    for i in range(len(notes)):
        index[notes[i]] = i 

    for track in list_track:
        # is_doublet = False
        # if (track[0], track[1]) in doublets:
        #     pi[12+doublets.index((track[0], track[1]))] += 1
        #     if (track[2],track[3]) in doublets:
        #         A[12+doublets.index((track[0],track[1]))][12+doublets.index((track[2],track[3]))]+=1
        #     else:
        #         A[12+doublets.index((track[0],track[1]))][track[2]]+=1
        #     is_doublet = True
        # else:
        if len(track)==0:
            print('allo')
            continue
        pi[index[track[0]]]+=1
        for i in range(len(track)-1):
        #     # if is_doublet==True:
        #     #     is_doublet=False
        #         # continue
        #     if (track[i], track[i+1]) in doublets:
        #         if i+2<len(track) and i+3<len(track) and (track[i+2],track[i+3]) in doublets:
        #             A[12+doublets.index((track[i], track[i+1]))][12+doublets.index((track[i+2],track[i+3]))]+=1
        #         else:
        #             if i+2<len(track):
        #                 A[12+doublets.index((track[i],track[i+1]))][track[i+2]]+=1
        #             else:
        #                 continue
        #         is_doublet=True
        #     else:
        #         if i+2<len(track) and (track[i+1],track[i+2]) in doublets:
        #             A[track[i]][12+doublets.index((track[i+1],track[i+2]))]+=1
        #         else:
            A[index[track[i]]][index[track[i+1]]]+=1
    n=np.linalg.norm(pi,ord=1)
    pi/=n
    for item in A:
        n=np.linalg.norm(item,ord=1)
        #print( n)
        item/=n
    print(sum(A[-1]))
    print(A)
    print(sum(A[0]))
    print(pi)
    nb_notes=int(np.mean(np.array([len(track) for track in list_track if len(track)!=0])))
    res=dict()
    res['A']=A
    res['pi']=pi
    res['velocity']=vel
    # res['temps']=temps
    res['nb_notes']=nb_notes
    res['index']=index
    # res['doublets']=doublets
    with open('markov_model.p','wb') as file:
        pkl.dump(res,file)
    #with open('markov_model.json','w') as file:
    #    json.dump(res,file)
    
notes, vel, nb_occ = extract(database)
#print(len(notes), vel, time, nb_occ)
#print(vel[list(vel.keys())[0]].keys())
print(len(vel))
#learn_markov_model(notes,vel)
t=time.time()
learn_markov_model_order(notes,vel)
print(time.time()-t)
    
