
import mido 
import os 

database = os.listdir("./database")

def convert_midibd(database):
    bd = []
    for m in database:
        try:
            bd.append(mido.MidiFile("./database/"+str(m)))
        except IOError:
            print('MThd not found. Probably not a MIDI file')   
            #print("./database/"+str(m))

    return bd 
        

def extract(database):
    res = []
    midibd = convert_midibd(database)
    for m in midibd:
        resm = []
        for track in m.tracks:
#            print('Track {}: {}'.format(i, track.name))
            for msg in track:
                if not msg.is_meta and msg.type == 'note_on':
                    resm.append(msg.note%12)
        res.append(resm)
                    
    return res 

def find_doublet(list_track,nb):
    count=dict()
    for i in range(12):
        for j in range(12):
            count[(i,j)]=0
    for track in list_track:
        for i in range(len(track)-1):
            count[(track[i],track[i+1])]+=1
    res=sorted(count.items(), key=lambda x: x[1],reverse=True)
    return res[:nb]

notes = extract(database)
print(len(notes))
res=find_doublet(notes,15)
print(res)
    