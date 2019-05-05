import mido
import numpy as np
import time
from extraction_fichiers import convert_midibd
import os
from mido import Message, MidiFile, MidiTrack


def extract_poly(database):
    res = []
    midibd = convert_midibd(database)
    velocity = dict()
    total = []
    current_chord=[]

    for m in midibd:
        resm=[]
        for track in m.tracks:
            for msg in track:

                if not msg.is_meta and (msg.type == 'note_on' or msg.type=='note_off'):
                    print(msg)
                    if msg.time<=2 and msg.velocity!=0:
                        current_chord.append(msg.note)
                    else:
                        if len(current_chord)!=0:
                            resm.append(tuple(current_chord))
                        current_chord=[]
                    print(current_chord)
                    #time.sleep(2)
    return resm

def write_midi_poly(list_chord,filename):#faut changer les valeurs de time et de velocity
    track=MidiTrack()
    outfile=MidiFile()
    outfile.tracks.append(track)
    track.append(Message('program_change', program=1))
    for chord in list_chord:
        track.append(Message('note_on',note=chord[0],velocity=70,time=120))
        for note in chord[1:]:
            track.append(Message('note_on',note=note,velocity=70,time=0))
        track.append(Message('note_on',note=chord[0],velocity=0,time=120))
        for note in chord[1:]:
            track.append(Message('note_on',note=note,velocity=0,time=0))
    outfile.save(filename)


if __name__=="__main__":
    database = os.listdir("./database")
    q=extract_poly(database)
    write_midi_poly(q,"coldplay.mid")