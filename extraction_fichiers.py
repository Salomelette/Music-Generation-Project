
import mido 
import os 

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
    time = dict()
    nb_occ = dict()
    
    for i in range(12):
        velocity[i] = 0
        time[i] = 0
        nb_occ[i] = 0
        
        
    for m in midibd:
        resm = []
        for track in m.tracks:
            for msg in track:
                if not msg.is_meta and msg.type == 'note_on':
                    note = msg.note%12 
                    resm.append(note)
                    velocity[note] += msg.velocity 
                    time[note] += msg.time
                    nb_occ[note] += 1 

        res.append(resm)
    
    for i in range(12):
        velocity[i] /= nb_occ[i]
        time[i] /= nb_occ[i]
                   
    return res, velocity, time, nb_occ

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

notes, vel, time, nb_occ = extract(database)
#print(len(notes), vel, time, nb_occ)
res=find_doublet(notes,20)
print(res)

    
