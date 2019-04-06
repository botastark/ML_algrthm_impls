#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:44:55 2019

@author: botaduisenbay

Dataset for linear classification 

"""

import numpy as np
import matplotlib.pyplot as plt
from random import randint

def line_to_illustrate(x_min,x_max,w):
    N=x_max-x_min
    x=np.array(range(x_min,x_max))
    X=np.transpose(np.array([np.ones(N),x]))
    W=np.transpose(np.asarray(w))
    Y=np.matmul(X, W)
    y_min=x_min*w[1]+w[0]
    y_max=x_max*w[1]+w[0]
    plt.plot(x,Y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    return W,y_min,y_max

def dataset_2classes(N,x1_min,x1_max,x2_min,x2_max,w ):
    x1=[]
    x2=[]
    for i in range(N):
        x1.append(randint(x1_min, x1_max))
        x2.append(randint(x2_min, x2_max))
    X=np.transpose(np.array([np.ones(N), np.asarray(x1, dtype=np.float32),np.asarray(x2, dtype=np.float32) ]))
    x_2=np.matmul(w,np.array([np.ones(N), np.asarray(x1, dtype=np.float32)]))
 
    Dataset=[]
    temp=[]
    for i in range(N) :
        temp=X[i][1:3].tolist()
        if x_2[i]>x2[i]:
            Dataset.append(temp + [0])
            plt.plot(x1[i], x2[i], 'bo')
        else:
            Dataset.append(temp + [1])
            plt.plot(x1[i], x2[i], 'rx')
    
    return Dataset
def LSM(D,N_samples):
    Data=[[1]+i for i in D]
    Data=np.asarray(Data)
    X=Data[:,0:3]
    Y=Data[:,3]
    T=t_vector(2,Y)
    X_t=np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X))
    W=np.matmul(X_t,T)
#    print(X_t.shape)
#    print(X.shape)
#    T_temp=np.asarray(T)
#    print(T_temp.shape)
    
#    X_temp=np.matmul(np.transpose(X_t),X)
#    print(X_temp.shape)

#    Out=np.matmul(np.transpose(T_temp),X_temp)
    Out=np.matmul(X,W)
    Class_temp=[np.where(i==np.max(i)) for i in Out]
    Y_out=[np.asscalar(i[0]) for i in Class_temp]
    counter=0
    for i in range(len(Y_out)):
        if Y[i]!=Y_out[i]:
            counter+=1
    print("LS method # of misclasified instances is ", str(counter))
                
    return X,W,Y_out


def t_vector(N_class, Y):
    result=[]

    Y=[int(i) for i in Y]
    for i in Y:
        temp=[0]*N_class
        temp[i]=1
        result.append(temp)
    return result
        

w=[-30, 3]
x1_min=-100
x1_max=100
W,x2_min,x2_max=line_to_illustrate(x1_min,x1_max,w)
N_samples=200
D=dataset_2classes(N_samples,x1_min,x1_max,x2_min,x2_max,w)

"""Least Squares Method"""
X,W_ls, Y_out=LSM(D,N_samples)
w1=W_ls[:,0]
w2=W_ls[:,1]
x=list(range(x1_min,x1_max))
w_ls=[(-w2[0]+w1[0])/2/w2[2], -w2[1]/w2[2]]
x2_ls=[w_ls[0]+w_ls[1]*i for i in x]
plt.plot(x,x2_ls,'m')
