# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:33:31 2016

@author: Tristan Le Nair & Florian Talour
"""

import numpy as np
#import matplotlib.pyplot as plt

# X0 contient une matrice 10 000 collones x 784 lignes
X0 = np.load('data/trn_img.npy')

# lbl0 contient les étiquettes de chacune des images parmi les chiffres de 0 à 9
# sous forme d'un tableau mono-dimensionnel 10 000 x 1
lbl0 = np.load('data/trn_lbl.npy')

# X1 contient une matrice 5 000 collones x 784 lignes
X1 = np.load('data/dev_img.npy')

# lbl1 contient les étiquettes de chacune des images parmi les chiffres de 0 à 9
# sous forme d'un tableau mono-dimensionnel 5 000 x 1
lbl1 = np.load('data/dev_lbl.npy')



def precision(nbExemplesMalClasses, nbTotalExemples):
    return nbExemplesMalClasses/nbTotalExemples 
    
def printErr(nbErreur, size):
	print("Taux erreur : {0}".format(precision(nbErreur*100.0, size)), '%')
    
# moyenne de chaque ligne
def moyenne(x):
    moy = np.zeros((10, x.shape[1]))
    for i in range(0, 10):
        moy[i] = np.average(x[lbl0 == i,:], axis=0)
    return moy    

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

# prediction de classes par la moyenne
def predictMoy(p):
	XRed0, XRed1, P = reduceMat(p)

	# entrainement
	moy = moyenne(XRed0)

	# definition des tableaux
	dist = np.zeros((10, X1.shape[0]))
	resultat = np.zeros(X1.shape[0], dtype = np.int)
	
	nberreur = 0
	# calcul du taux d'erreur
	for i in range(0, 10):
		temp = np.subtract(XRed1,moy[i])*np.subtract(XRed1,moy[i])
		dist[i] = np.sum(temp, axis=1)
            
	resultat = np.argmin(dist, axis = 0)
	nberreur = sum(resultat != lbl1)

	# affichage du taux d'erreur
	printErr(nberreur, XRed1.shape[0])
	return precision(nberreur*100, XRed1.shape[0]), resultat

# prediction de classes par le plus proche voisin
def predictPPV(p):
   	
    # calcul de la reduction de X0 et X1 pour les obtenir en dimension p
    XRed0, XRed1, P = reduceMat(p)
    
    # soit un sous-ensemble de la base d'apprentissage
    #XRed0 = trainPPV(XRed0,p) # on enlève p valeurs à X0
    
    # definition des tableaux
    dist = np.zeros((X1.shape[0], X0.shape[0]))
    resultat = np.zeros(X1.shape[0], dtype = np.int)
    
    nberreur = 0
    
    dist = np.zeros((X0.shape[0], X1.shape[0]))
    
    # calcul du taux d'erreur
    for i in range(XRed0.shape[0]):
        temp = np.subtract(XRed1, XRed0[i])*np.subtract(XRed1, XRed0[i])
        dist[i] = np.sum(temp, axis = 1)
    
            
    resultat = lbl0[np.argmin(dist, axis = 0)]
    nberreur = sum(resultat != lbl1)
            
    # affichage du taux d'erreur
    printErr(nberreur, XRed1.shape[0])
    
    return precision(nberreur*100, XRed1.shape[0]), resultat

def confmat(true, pred):
    dim = max(pred)+1
    z = dim*true + pred
    zcount = np.bincount(z, minlength = dim*dim)
    
    print(zcount.reshape(dim,dim))

def main():  
    # DMIN avec ACP :
    #res, prediction = predictMoy(100)
    #confmat(lbl1, prediction)

    # 1PPV avec 1 ACP
    res, prediction = predictPPV(100)
    confmat(lbl1, prediction)

    # 1PPV avec plusieurs ACP
    #tabPrecision = np.zeros(5000/100)
    #for i in range(100, 5000, 500):
    #    tabPrecision[i/100] = predictPPV(i)
   
    #plt.title("Taux d'erreur de detection du chiffre en fonction du nombre de vecteurs conservé")
    #plt.plot(tabPrecision)
    #plt.ylabel("Taux d'erreur")
    #plt.xlabel("Taille du vecteur choisi")
    #plt.show()

	# Sauvegarde du meilleur pourcentage
	#np.save('test-1nn', res)
    
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


    
    
    
    
