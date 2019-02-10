from mido import Message, MidiFile, MidiTrack
import mido


mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

track.append(Message('program_change', program=12, time=0))
for i in range(40):
	track.append(Message('note_on', note=64, velocity=64, time=100))
	track.append(Message('note_off', note=64, velocity=127, time=100))

mid.save('new_song.mid')

