import mido
import numpy as np
import time
from extraction_fichiers import convert_midibd
import os
from mido import Message, MidiFile, MidiTrack
import pickle as pkl
from collections import Counter

range_note=127
split=0.5

def extract_poly(database):
    res = []
    midibd = convert_midibd(database)
    velocity = dict()
    total = []
    current_chord=[]
    ref_chord=np.zeros(range_note)
    for m in midibd:
        deb=np.zeros(range_note)
        deb[0]=-1
        resm=[]
        #resm=[deb]
        for track in m.tracks:
            for msg in track:

                if not msg.is_meta and (msg.type == 'note_on' or msg.type=='note_off'):
                    #print(msg)
                    if msg.time<=2 and msg.velocity!=0:
                        current_chord.append(msg.note)
                    else:
                        if len(current_chord)!=0:
                            ref_chord[current_chord]=1
                            resm.append(ref_chord)
                            ref_chord=np.zeros(range_note)
                        current_chord=[]
                    #print(current_chord)
                    #time.sleep(2)
        end=np.zeros(range_note)
        end[-1]=-1
        #resm.append(end)
        res.append(np.array(resm))

    return res

def extract_poly2(database):
    try:
        with open("database_poly.p",'rb') as file:
            res,_=pkl.load(file)
        print("fichier trouvé")
        return res
    except:
        pass
    res = []
    midibd = convert_midibd(database)
    velocity = dict()
    total = []
    current_chord=[]

    ref_chord=np.zeros(2*range_note+2)
    k=0
    for m in midibd:
        resm=[]
        temps=0
        for track in m.tracks:
            for msg in track:
                if not msg.is_meta and (msg.type == 'note_on' or msg.type=='note_off'):
                    resm.append({'type':msg.type,"note":msg.note,'time':msg.time,'velocity':msg.velocity})
        #print(resm)
        duree=sum([item['time'] for item in resm])
        #print(duree)
        
        duree+=split-duree%split
        #dic={k:i for i,k in enumerate(range(0,duree+split+1,split))}
        #print(duree/split)
        mat=np.zeros(2*range_note+2).reshape(1,2*range_note+2)
        temps=0
        i=0
        #print(dic)
        """for item in resm:
            temps+=item['time']
            print(item)
            if item['type']=='note_on':
                mat[dic[split*round(temps,split)]][item['note']]=1
            else:
                mat[dic[split*round(temps,split)]][range_note+item['note']]=1"""

        
        for item in resm:
            #print(item)

            bound=temps+mido.tick2second(item['time'],m.ticks_per_beat,500000)
            #print("temps=",temps)
            #print("bound=",bound)
            kk=0
            bb=False
            while temps<bound:
                #print(temps)
                temps+=split
                i+=1
                kk+=1
                bb=True
                mat=np.vstack((mat,np.zeros(2*range_note+2)))
            if bb:
                k+=kk-1
            #print(mat.shape)
            #print(i)
            if item['type']=='note_off':
                mat[i][range_note+item['note']]=1
            elif item['velocity']==0:
                mat[i][range_note+item['note']]=1
            else:
                mat[i][item['note']]=1
                if item['note'] not in velocity.keys():
                    velocity[item['note']]=Counter()
                velocity[item['note']][item['velocity']] += 1

        bof=np.zeros(2*range_note+2)
        eof=np.zeros(2*range_note+2)
        bof[-2]=1
        eof[-1]=1
        mat=np.vstack((bof,mat,eof))
        res.append(mat)
        print(len(res[-1]))
    print("nb accord vide:",k)
    for item in velocity.keys():
        total=sum([k for k in velocity[item].values()])
        for key in velocity[item]:
            velocity[item][key]/=total
    with open("database_poly.p",'wb') as file:
        pkl.dump((res,velocity),file)
    return res

def write_midi_poly(list_chord,filename):#faut changer les valeurs de time et de velocit
    track=MidiTrack()
    outfile=MidiFile()
    outfile.tracks.append(track)
    track.append(Message('program_change', program=1))
    for ref_chord in list_chord:
        chord=np.where(ref_chord==1)[0]
        if len(chord)==0:
            continue
        track.append(Message('note_on',note=chord[0],velocity=70,time=120))
        for note in chord[1:]:
            track.append(Message('note_on',note=note,velocity=70,time=0))
        track.append(Message('note_on',note=chord[0],velocity=0,time=150))
        for note in chord[1:]:
            track.append(Message('note_on',note=note,velocity=0,time=0))
    outfile.save(filename)

def write_midi_poly2(list_chord,filename):#faut changer les valeurs de time et de velocit
    with open("database_poly.p",'rb') as file:
            _,velocity=pkl.load(file)
    track=MidiTrack()
    outfile=MidiFile()
    outfile.tracks.append(track)
    track.append(Message('program_change', program=1))
    add_time=0
    for ref_chord in list_chord:
        on=ref_chord[:range_note]
        off=ref_chord[range_note:-2]
        #print(len(on))
        #print(len(off))
        on_chord=np.where(on==1)[0]
        off_chord=np.where(off==1)[0]
        #print(on_chord)
        #print(off_chord)
        #if len(chord)==0:
        #    pass
        #    continue

        if len(off_chord)!=0:
            track.append(Message('note_off',note=off_chord[0],velocity=0,time=int(mido.second2tick(split+add_time,200,500000))))
            for note in on_chord[:]:
                velo_key=list(velocity[note].keys())
                velo_values=list(velocity[note].values())
                tirage=np.random.choice(len(velo_key), 1, p=velo_values)[0]
                vel=velo_key[tirage]
                track.append(Message('note_on',note=note,velocity=vel,time=0))
            for note in off_chord[1:]:
                track.append(Message('note_off',note=note,velocity=0,time=0))
            add_time=0
        elif len(on_chord)!=0:
            note=on_chord[0]
            velo_key=list(velocity[note].keys())
            velo_values=list(velocity[note].values())
            tirage=np.random.choice(len(velo_key), 1, p=velo_values)[0]
            vel=velo_key[tirage]
            track.append(Message('note_on',note=on_chord[0],velocity=vel,time=int(mido.second2tick(split+add_time,200,500000))))
            for note in on_chord[1:]:
                velo_key=list(velocity[note].keys())
                velo_values=list(velocity[note].values())
                tirage=np.random.choice(len(velo_key), 1, p=velo_values)[0]
                vel=velo_key[tirage]
                track.append(Message('note_on',note=note,velocity=vel,time=0))
                add_time=0
        else:
            add_time+=split
    outfile.save(filename)

if __name__=="__main__":
    database = os.listdir("./database")
    q=extract_poly2(database)
    #print(q)
    write_midi_poly2(q[6],"yanntest.mid")