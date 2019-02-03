
import mido 
import os 

database = os.listdir("/home/salom/ProjetM1/database")

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


notes = extract(database)
print(len(notes))
        
    