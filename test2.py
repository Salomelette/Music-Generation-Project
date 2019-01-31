from mido import Message, MidiFile, MidiTrack
import mido
"""
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=12, time=0))
track.append(Message('note_on', note=64, velocity=64, time=32))
track.append(Message('note_off', note=64, velocity=127, time=32))

mid.save('new_song.mid')

"""
with open("filename.tx",'r') as file:
    data=file.read().split('\n')

mid=mido.MidiFile("MIDI_sample.mid")
mid=mido.MidiFile("./database/{}".format(data[1]))
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))

    for msg in track:
    	print(msg)
    	if msg.is_meta:
    		print(msg.type)