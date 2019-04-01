#https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

#https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from extraction_fichiers import extract
import os
database = os.listdir("./database")
notes, vel, nb_occ = extract(database)

def prepare_sequence(seq, to_int):
    idxs = [to_int[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class Rnn(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size):
        super(Rnn, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeds.view(len(sequence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sequence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    

model=Rnn(100,1,len(nb_occ))
loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)



notes_to_int=dict()
for item in nb_occ.keys():
    if item not in notes_to_int.keys():
        notes_to_int[item]=len(notes_to_int)

sequence_length = 100
network_input = []
network_output = []
for track in notes:
    for i in range(0, len((track)) - sequence_length, 1):
        sequence_in = track[i:i + sequence_length]
        sequence_out = track[i + sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)

with torch.no_grad():
    inputs = prepare_sequence(notes[0], notes_to_int)
    tag_scores = model(inputs)
    print(tag_scores)
   
training_data=[(network_input[i],network_output[i]) for i in range(len(network_input))]

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence,  notes_to_int)
        targets = prepare_sequence([tags], notes_to_int)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        print(tag_scores.shape)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        print(targets.shape)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()