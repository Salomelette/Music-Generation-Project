import json
import numpy as np
import mido
import time
import random



def generate_music():
    with open('markov_model.json','r') as file:
        data=file.read()
    data=json.loads(data)
    print(data)
    A=data['A']
    pi=data['pi']
    vel=data['velocity']
    temps=data['temps']
    nb_notes=data['nb_notes']
    
    
    
    
    
    
generate_music()