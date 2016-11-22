# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:33:31 2016

@author: tristan & florian
"""

import numpy as np
import matplotlib.pyplot as plt

# X0 contient une matrice 10 000 collones x 784 lignes
X0 = np.load('trn_img.npy')

# lbl0 contient les étiquettes de chacune des images parmi les chiffres de 0 à 9
# sous forme d'un tableau mono-dimensionnel 10 000 x 1
lbl0 = np.load('trn_lbl.npy')

# X1 contient une matrice 5 000 collones x 784 lignes
X1 = np.load('dev_img.npy')

# lbl1 contient les étiquettes de chacune des images parmi les chiffres de 0 à 9
# sous forme d'un tableau mono-dimensionnel 5 000 x 1
lbl1 = np.load('dev_lbl.npy')

# moyenne de chaque ligne
def moyenne(x):
    moy = np.zeros((10, x.shape[1]))
    for i in range(0, 10):
        moy[i] = np.average(x[lbl0 == i,:], axis=0)
    return moy

def precison(nbExemplesMalClasses, nbTotalExemples):
    return nbExemplesMalClasses/nbTotalExemples 
    
# 1PPV obtenir un sous-ensemble de la base d'apprentissage, on enlève p valeurs à x
def trainPPV(x, p):
    return x[:,:p] 

# ACP
def reduceMat(p):

    C = np.cov(X0, rowvar=0)
    v,w = np.linalg.eigh(C)

    P = -w[:, len(w)-p : len(w)] 
    
    #X0_prime = np.multiply(X0, v)
    #X1_prime = np.multiply(X1, v)
    
    XRed0 = np.matmul(X0, P)
    XRed1 = np.matmul(X1, P)

    return XRed0, XRed1, P

def predictMoy(p):
    XRed0, XRed1, P = reduceMat(p)

    # entrainement
    moy = moyenne(XRed0)

    # definition des tableaux
    dist = np.zeros((X1.shape[0], 10))
    resultat = np.zeros(X1.shape[0])
    
    nberreur = 0
    # calcul du taux d'erreur
    for j in range(XRed1.shape[0]):
		for i in range(0, 10):
			dist[j][i] = np.sum(np.subtract(XRed1[j],moy[i])*np.subtract(XRed1[j],moy[i]))
		
		resultat[j] = np.argmin(dist[j], axis=0)

		if resultat[j] != lbl1[j]:
				nberreur = nberreur + 1
		   
	# affichage du taux d'erreur
	print ("Taux erreur :", precison(nberreur*100.0, XRed1.shape[0]), "%")
	return precison(nberreur*100.0, XRed1.shape[0])
            
def predictPPV(p):
   	
    # calcul de la reduction de X0 et X1 pour les obtenir en dimension p
    XRed0, XRed1, P = reduceMat(p)
    
    # soit un sous-ensemble de la base d'apprentissage
    #XRed0 = trainPPV(XRed0,p) # on enlève p valeurs à X0
    
    # definition des tableaux
    dist = np.zeros((X1.shape[0], X0.shape[0]))
    resultat = np.zeros(X1.shape[0])
    
    nberreur = 0
    
    # calcul du taux d'erreur
    for j in range(XRed1.shape[0]):
        for i in range(XRed0.shape[0]):
            dist[j][i] = np.sum(np.subtract(XRed1[j],XRed0[i])*np.subtract(XRed1[j],XRed0[i]))
        
        resultat[j] = lbl0[np.argmin(dist[j], axis=0)]
        
        if resultat[j] != lbl1[j]:
            nberreur = nberreur + 1
            
    # affichage du taux d'erreur
    print ("Taux erreur :", precison(nberreur*100.0, XRed1.shape[0]), "%")
    
    return precison(nberreur*100.0, XRed1.shape[0])

def main():    

	# DMIN avec ACP :
	#res = predictMoy(100)

	# 1PPV avec 1 ACP
	res = predictPPV(100)

	# 1PPV avec plusieurs ACP
	#tabPrecision = np.zeros(5000/100)
	#for i in range(100, 5000, 500):
	#    tabPrecision[i/100] = predictPPV(i)

	#plt.title("Taux d'erreur de detection du chiffre en fonction du nombre de vecteurs conservé")
	#plt.plot(tabPrecision)
	#plt.ylabel("Taux d'erreur")
	#plt.xlabel("Taille du vecteur choisi")
	#plt.show()

	np.save('test-1nn', res)
	# Afficher une image individuellement
	#img = XRed0[0].reshape(p/10,10)
	#plt.imshow(img, plt.cm.gray)
	#plt.show()

	# Afficher une image individuellement dans sa dimension initiale 28 x 28
	#img = X0[0].reshape(28,28)
	#plt.imshow(img, plt.cm.gray)
	#plt.show()

#fonction main
if __name__ == "__main__":
    main()   


    
    
    
    
