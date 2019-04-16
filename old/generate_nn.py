
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from extraction_fichiers import extract
from music_nn import Rnn
import numpy as np
import os
from discord_hooks import Webhook

from generateur import create_midi_file

database = os.listdir("./database")
notes, vel, nb_occ = extract(database)
dd='cuda'
#dd='cpu'
cuda0 = torch.device(dd)

def prepare_sequence(seq, to_int):
    idxs = [to_int[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long,device=cuda0)

model=Rnn(128,128,len(nb_occ))
if dd=='cuda':
    model.cuda()
    loss_function = nn.CrossEntropyLoss().cuda()
else:
    loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)



PATH="lstm_model.model"
model.load_state_dict(torch.load(PATH))
model.eval()

int_to_notes=dict()
notes_to_int=dict()
for item in nb_occ.keys():
    if item not in notes_to_int.keys():
        notes_to_int[item]=len(notes_to_int)
        int_to_notes[notes_to_int[item]]=item

sequence_length = 10
network_input = []
network_output = []
for track in notes:
    for i in range(0, len((track)) - sequence_length, 1):
        sequence_in = track[i:i + sequence_length]
        sequence_out = track[i+1:i + sequence_length+1]
        network_input.append(sequence_in)
        network_output.append(sequence_out)


training_data=[(network_input[i],network_output[i]) for i in range(len(network_input))]


res_int=[]
inputs = prepare_sequence(training_data[0][0], notes_to_int)
res_int+=inputs.tolist()
with torch.no_grad():
    
    for i in range(300):
        
        tag_scores = model(inputs)
        indice=tag_scores[-1].argmax()
        res_int+=[indice]
        #resinputs=prepare_sequence(res_int[-10:-1],notes_to_int) 
        inputs=torch.tensor(res_int[len(res_int)-10:], dtype=torch.long,device=cuda0)
        
        #print(tag_scores)
        #print(res_int)

res=[int_to_notes[int(i)] for i in res_int]

index_inv=int_to_notes
res2=[]
for i in range(len(res)):
    #↨print(i)
    velo_keys=list(vel[res[i]].keys())
    velo_values=list(vel[res[i]].values())
    tirage=np.random.choice(len(velo_keys), 1, p=velo_values)[0]

    res2.append((res[i][0],res[i][1],velo_keys[tirage]))
filename="allo3.mid"
create_midi_file(res2,filename)