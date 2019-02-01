
import mido 
import os 

database = os.listdir("/home/salom/ProjetM1/database")
print(database[-1])

def convert_midibd(database):
    bd = []
    for m in database:
        try:
            bd.append(mido.MidiFile("./database/"+str(m)))
        except IOError:
            print('MThd not found. Probably not a MIDI file')   
            print("./database/"+str(m))

    return bd 
        

def extract(database):
    res = []
    midibd = convert_midibd(database)
    for m in midibd:
        for i, track in enumerate(m.tracks):
#            print('Track {}: {}'.format(i, track.name))
            for msg in track:
                midibd.append(msg.note%12)
    return res 


notes = extract(database)
print(notes)
        
    