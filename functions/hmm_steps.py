# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:43:19 2020

@author: flach
"""

import numpy as np
import matplotlib.pyplot as plt

def viterbi(A, B, O):
    N = A.shape[0]-2
    d = np.zeros(len(O)*N).reshape(N,len(O))
    psi = np.zeros(len(O)*N).reshape(N,len(O))
    h = np.zeros(N)
    z = np.zeros(len(O))
    for j in np.arange(N):
        d[j,0] = A[0,j+1]*B[j,O[0]]
        psi[j,0] = 0
        
    for t in np.arange(1, len(O)):
        for j in np.arange(N):
            for i in np.arange(N):
                h[i] = d[i,t-1]*A[i+1,j+1]
            d[j,t] = max(h)*B[j,O[t]]
            psi[j,t] = np.argmax(h)
    
    eval_max = np.max(d[:,len(O)-1])
    z[len(O)-1] = np.argmax(d[:,len(O)-1])
        
    for t in np.arange(len(O)-1,0,-1):
        z[t-1] = psi[int(z[t]),t]
    return d, eval_max, z

def hmm_prod(A, B, O):
    N = A.shape[0]-2
    #d = np.zeros(len(O)*N).reshape(N,len(O))
    h = np.zeros(N)
    ev = np.zeros(O.shape[0])
    for m in range(O.shape[0]):
        d = np.zeros(len(O[m])*N).reshape(N,len(O[m]))
        for j in range(N):
            d[j,0] = A[0,j+1]*B[j,O[m][0]]
        for t in np.arange(1, len(O[m])):
            for j in np.arange(N):
                for i in np.arange(N):
                    h[i] = d[i,t-1]*A[i+1,j+1]
                d[j,t] = sum(h)*B[j,O[m][t]]   
        ev[m] = np.sum(d[:,len(O[m])-1])
    eval_hmm = np.sum(ev)/O.shape[0]    
    return d, eval_hmm

def forward(A, B, O):
    n_z = A.shape[0]-2
    d = np.zeros(len(O)*n_z).reshape(n_z,len(O))
    T = len(O)
    for i in range(n_z):
        d[i,0] = A[0,i+1]*B[i,O[0]]
    for t in range(1, T):
        for j in range(n_z):
            d[j,t] = B[j,O[t]] * np.sum(d[:,t-1] * A[1:n_z+1,j+1]) 
    pw = np.sum(d[:,d.shape[1]-1])
    return d, pw

def backward(A, B, O):
    n_z = A.shape[0]-2
    d = np.zeros((len(O))*n_z).reshape(n_z,len(O))
    T = len(O)
    for i in np.arange(n_z):
        #d[i,T-1] = A[i+1,n_z+1]
        d[i,T-1] = 1

    for t in np.arange(T-2,-1,-1):
        for j in np.arange(n_z):
            d[j,t] = np.sum(B[:, O[t+1]] * d[:,t+1] * A[j+1, 1:n_z+1])
    pw = np.sum(A[0,1:n_z+1] * B[:, O[0]] * d[:,0])
    return d, pw

def gamma(alpha, beta):
    pw = np.sum(alpha[:,alpha.shape[1]-1])
    gm = alpha*beta/pw
    return gm

def xi(A, B, alpha, beta, O):
    T = len(O)
    n_z = alpha.shape[0]
    pw = np.sum(alpha[:,alpha.shape[1]-1])
    x = np.zeros(n_z*n_z*T).reshape(n_z,n_z,T)
    Axi = A[1:n_z+1, 1:n_z+1]
    for t in np.arange(0,T-1):
        for i in range(n_z):
            x[i,:,t] = alpha[i,t]*Axi[i,:]*B[:,O[t+1]]*beta[:,t+1]/pw
    for i in range(n_z):
        x[i,:,T-1] = alpha[i,T-1]*Axi[i,:]/pw
    return x
    
def hmm_init():
    n_z = 2
    n_o = 3
    A = np.array([0, 1, 0, 0, 0, 0.8, 0.2, 1, 0, 0.6, 0.4, 1, 0, 0, 0, 0]).reshape(n_z+2,n_z+2)
    B = np.array([0.6, 0.3, 0.1, 0.2, 0.3, 0.5]).reshape(n_z,n_o)
    return A, B 

def hmm_rand_init(n_z, n_o):
    np.random.seed(5)
    A = np.zeros((n_z+2)*(n_z+2)).reshape(n_z+2,n_z+2)
    A[:n_z+1,1:n_z+1] = np.random.rand((n_z+1)*(n_z)).reshape(n_z+1,n_z)
    for i in range(n_z+1):
        A[i,:] = A[i,:]/np.sum(A[i,1:])
        A[i,n_z+1] = 1
    B = np.random.rand(n_z*n_o).reshape(n_z,n_o)
    for i in range(n_z):
        B[i,:] = B[i,:]/np.sum(B[i,:])
    return A, B 


def calc_AB(N, S, A, B, O):
    n_z = N
    n_o = S
    A_neu = np.zeros((n_z+2, n_z+2))
    B_neu = np.zeros((n_z, n_o))
    B_h = np.zeros((O.shape[0], n_z, n_o))
    A_a = np.zeros((n_z, O.shape[0]))
    A_e = np.zeros((n_z, O.shape[0]))
    xi_hz = np.zeros((O.shape[0], n_z, n_z))
    xi_hn = np.zeros((O.shape[0], n_z, n_z))
    pw = np.zeros(O.shape[0])
    b_h = np.zeros((O.shape[0], n_z))
    for m in range(O.shape[0]):
        alpha, b = forward(A, B, O[m])
        beta, b = backward(A, B, O[m])
        gam = gamma(alpha, beta)
        x = xi(A, B, alpha, beta, O[m])
        # Anfangszustandswahrscheinlichkeiten
        A_a[:,m] = gam[:,0]
    
        # Transitionswahrscheinlichkeiten
        for i in range(n_z):
            pw[m] = np.sum(alpha[:,len(O[m])-1])
            for j in range(n_z):
                xi_hz[m,i,j] = np.sum(x[i,j,0:len(O[m])])
                xi_hn[m,i,j] = np.sum(gam[i,:])
        #A[1:n_z+1,1:n_z+1] = xi_h

        # Endzustandswahrscheinlichkeiten
        for i in range(n_z):
            A_e[i,m] = gam[i,gam.shape[1]-1]/sum(gam[i,:])
        #A_e[:,m] = gam[:,gam.shape[1]-1]
        
        # Emissionswahrscheinlichkeiten
        for l in range(n_o):
            res_list = [i for i in range(len(O[m])) if O[m][i] == l]
    
            for j in range(n_z):
                b_h[m,j] = np.sum(gam[j,:])
                B_h[m,j,l] = np.sum(gam[j,res_list]) 
    for i in range(n_z):
        A_neu[0,i+1] = np.sum(A_a[i,:])/O.shape[0]
        for j in range(n_z):
            A_neu[i+1,j+1] = np.sum(xi_hz[:,i,j])/np.sum(xi_hn[:,i,j])
        #A_neu[i+1,n_z+1] = np.sum(A_e[i,:])/O.shape[0]
    for i in range(n_z):
        for j in range(n_o):
            B_neu[i,j] = np.sum(B_h[:,i,j])/np.sum(b_h[:,i])
    return A_neu, B_neu

def plot_res(res):
    plt.plot(res)
    plt.title('Lernkurve')
    plt.xlabel('Iterationen')
    plt.ylabel('Produktionswahrscheinlichkeit')
    plt.grid()
    

def mod(O, N, S, n_iter):
    A, B = hmm_rand_init(N, S)
    eps = 1e-20
    res = np.zeros(n_iter)
    i = 0
    pw_old = eps
    while (i < n_iter):
        A, B = calc_AB(N, S, A, B, O)
        trel, pw = hmm_prod(A, B, O)
        res[i] = pw
        i = i+1
        if pw - pw_old < eps:
            break
        pw_old = pw
    return A, B

