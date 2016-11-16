# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:33:31 2016

@author: tristan
"""

import numpy as np

#X0 contient une matrice 10 000 x 784
X0 = np.load('/home/tristan/Documents/Polytech/Github/ProjetBigData/trn_img.npy')

#lbl0 contient les étiquettes de chacune des images parmi les chiffresde 0 à 9
#sous forme d'un tableau mono-dimensionnel 10 000 x 1
lbl0 = np.load('/home/tristan/Documents/Polytech/Github/ProjetBigData/trn_lbl.npy')

X1 = np.load('/home/tristan/Documents/Polytech/Github/ProjetBigData/dev_img.npy')

lbl1 = np.load('/home/tristan/Documents/Polytech/Github/ProjetBigData/dev_lbl.npy')

#moyenne de chaque ligne
def moyenne(x, nbvalues):
    moy = np.zeros((10,nbvalues))
    
    for i in range(0, 10):
        moy[i] = np.average(x[lbl0 == i,:], axis=0)
    return moy

def precison(nbExemplesMalClasses, nbTotalExemples):
    return nbExemplesMalClasses/nbTotalExemples 
    
#1PPV
def trainPPV(x, p):
    return x[p:,:] 

#ACP
def reduceMat(X0, X1):
    p = 100

    C = np.cov(X0, rowvar=0)
    v,w = np.linalg.eigh(C)

    P = -w[:, len(w)-p : len(w)] 
    
    #X0_prime = np.multiply(X0, v)
    #X1_prime = np.multiply(X1, v)
    
    XRed0 = np.matmul(X0, P)
    XRed1 = np.matmul(X1, P)

    return P, XRed0, XRed1, p

P, XRed0, XRed1, p = reduceMat(X0, X1)
moy = moyenne(XRed0, p)

def predictMoy():
    cpt=0
    nberreur = 0
    dist = np.zeros((X1.shape[0], 10))
    resultat = np.zeros(X1.shape[0])
    
    for j in range(XRed1.shape[0]):
        for i in range(0, 10):
            dist[j][i] = np.sum(np.subtract(XRed1[j],moy[i])*np.subtract(XRed1[j],moy[i]))
        
        resultat[j] = np.argmin(dist[j], axis=0)
        
        if resultat[j] != lbl1[j]:
            nberreur = nberreur + 1
            
        cpt += 1

def predictPPV():
    cpt=0
    nberreur = 0
    Xt = trainPPV(X0,100)
    dist = np.zeros((X1.shape[0], Xt.shape[0]))
    resultat = np.zeros(X1.shape[0])
    
    for j in range(X1.shape[0]):
        for i in range(Xt.shape[0]):
            dist[j][i] = np.sum(np.subtract(X1[j],Xt[i])*np.subtract(X1[j],Xt[i]))
        
        resultat[j] = np.argmin(dist[j], axis=0)
        
        if resultat[j] != lbl1[j]:
            nberreur = nberreur + 1
            
        cpt += 1
    return nberreur

#P, XRed0, XRed1, p = reduceMat(X0, X1)
#moy = moyenne(XRed0, p)
err = predictPPV()
    
#Afficher une image individuellement dans sa dimension initiale 10 x 10
import matplotlib.pyplot as plt
img = XRed0[0].reshape(p/10,10)
plt.imshow(img, plt.cm.gray)
plt.show()

#Afficher une image individuellement dans sa dimension initiale 28 x 28
img = X0[0].reshape(28,28)
plt.imshow(img, plt.cm.gray)
plt.show()    
    
print ("Taux erreur :", precison(err*100.0, X1.shape[0]), "%")



    
    
    
    