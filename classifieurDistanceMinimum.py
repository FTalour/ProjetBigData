# -*- coding: utf-8 -*-
"""Classifieur à distance Minimum (DMIN)"""

"""Lecture des donnees"""

import numpy as np

#X0 contient une matrice 10 000 x 784
X0 = np.load('/home/tristan/Documents/Polytech/Github/ProjetBigData/trn_img.npy')

#lbl0 contient les étiquettes de chacune des images parmi les chiffresde 0 à 9
#sous forme d'un tableau mono-dimensionnel 10 000 x 1
lbl0 = np.load('/home/tristan/Documents/Polytech/Github/ProjetBigData/trn_lbl.npy')

X1 = np.load('/home/tristan/Documents/Polytech/Github/ProjetBigData/dev_img.npy')

lbl1 = np.load('/home/tristan/Documents/Polytech/Github/ProjetBigData/dev_lbl.npy')
    
#covariance = np.cov(moy)
#diagonale = np.diag(covariance)
    
def dstMahalanobis(x, mu):
    epsilon = np.zeros(len(x))
    epsilon = epsilon + ((x-mu)*((x-mu).T))/len(x)
    epsilonInv =  np.linalg.inv(epsilon)
    return np.transpose(x-mu)*epsilonInv*(x-mu)


#moyenne de chaque ligne
def moyenne(x, nbvalues):
    moy = np.zeros((10,nbvalues))
    for i in range(0, 10):
        moy[i] = np.average(x[lbl0 == i,:], axis=0)
    return moy
   

#Performance : Taux erreur
def precison(nbExemplesMalClasses, nbTotalExemples):
    return nbExemplesMalClasses/nbTotalExemples   
   
moy = moyenne(X0, 784)

#train():
#guess(): 

#for j in range(X1.shape[0]):
cpt = 0
nberreur = 0
dist = np.zeros((X1.shape[0], 10))
resultat = np.zeros(X1.shape[0])

for j in X1:
    for i in range(0, 10):
        dist[cpt][i] = np.sum(np.subtract(X1[cpt],moy[i])*np.subtract(X1[cpt],moy[i]))
    resultat[cpt] = np.argmin(dist[cpt], axis=0)
    if resultat[cpt] != lbl1[cpt]:
        nberreur = nberreur + 1
    cpt += 1
print precison(nberreur*100.0, X1.shape[0])


#Afficher une image individuellement dans sa dimension initiale 28 x 28
import matplotlib.pyplot as plt
img = X0[0].reshape(28,28)
plt.imshow(img, plt.cm.gray)
plt.show()
