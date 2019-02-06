import json
import numpy as np
import mido


def generate_music():
    with open('markov_model.json','r') as file:
        data=file.read()
    data=json.loads(data)
    A=data['A']
    pi=data['pi']
    vel=data['velocity']
    temps=data['temps']
    nb_notes=data['nb_notes']
    nb_notes=10
    doublets=data['doublets']
    set_notes=list(range(12))+doublets
    print(set_notes)
    print(len(set_notes))
    is_doublet=False
    cs=np.cumsum(pi)
    a=np.random.rand(1)
    i=0
    while cs[i]<a:
        i+=1
    old_val=i
    res=[old_val]
    if old_val>=12:
        is_doublet=True
    for j in range(nb_notes-1):
        if is_doublet:
            is_doublet=False
            continue
        cs=np.cumsum(A[old_val])
        a=np.random.rand(1)
        i=0
        while cs[i]<a:
            i+=1
        old_val=i
        if old_val>=12:
            is_doublet=True
        res.append(old_val)

    print(res)
    
    
generate_music()