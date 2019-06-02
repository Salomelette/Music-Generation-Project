# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 08:02:30 2019

@author: Xavier
"""
import numpy as np
import time
def compare(seq1,seq2):
    seq1.insert(0,"BOT")
    seq2.insert(0,"BOT")
    mat=np.zeros(len(seq1)*len(seq2)).reshape(len(seq1),len(seq2))
    for i in range(1,mat.shape[0]):
        for j in range(1,mat.shape[1]):
#            print("i=",i," j=",j)
#            print("seq1:",seq1[i])
#            print("seq2:",seq2[j])
            if seq1[i]==seq2[j]:
                mat[i][j]=mat[i-1][j-1]+1
            else:
                mat[i][j]=max(mat[i][j-1],mat[i-1][j])
    return mat.max()

def check_sub_seq(base,seq):
    t=time.time()
    stock=0
    for track in base:
        stock=max(compare(track,seq),stock)
    print(time.time()-t)
    return stock