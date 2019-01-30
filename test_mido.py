import mido

mid=mido.MidiFile("MIDI_sample.mid")
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))

    for msg in track:
    	if msg.is_meta:
    		print(msg.type)
