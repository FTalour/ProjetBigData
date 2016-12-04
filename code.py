# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:33:31 2016

@author: Tristan Le Nair & Florian Talour
"""

import numpy as np
import matplotlib.pyplot as plt

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
def reduceMat(p, trainMat=X0, devMat=X1):

    C = np.cov(trainMat, rowvar=0)
    v,w = np.linalg.eigh(C)

    P = -w[:, len(w)-p : len(w)] 
    
    #trainMat_prime = np.multiply(trainMat, v)
    #devMat_prime = np.multiply(devMat, v)
    
    XRed0 = np.matmul(trainMat, P)
    XRed1 = np.matmul(devMat, P)

    return XRed0, XRed1, P

# prediction de classes par la moyenne
def predictMoy(trainMat=X0, devMat=X1):

    # entrainement
    moy = moyenne(trainMat)

    # definition des tableaux
    dist = np.zeros((10, devMat.shape[0]))
    resultat = np.zeros(devMat.shape[0], dtype = np.int)

    nberreur = 0
    # calcul du taux d'erreur
    for i in range(0, 10):
        temp = np.subtract(devMat,moy[i])*np.subtract(devMat,moy[i])
        dist[i] = np.sum(temp, axis=1)
            
    resultat = np.argmin(dist, axis = 0)
    nberreur = sum(resultat != lbl1)

    # affichage du taux d'erreur
    printErr(nberreur, devMat.shape[0])
    return precision(nberreur*100, devMat.shape[0]), resultat

# prediction de classes par le plus proche voisin
def predictPPV(trainMat=X0, devMat=X1):
    
    # definition des tableaux
    dist = np.zeros((devMat.shape[0], trainMat.shape[0]))
    resultat = np.zeros(devMat.shape[0], dtype = np.int)
    
    nberreur = 0
    
    # calcul du taux d'erreur
    for i in range(devMat.shape[0]):
        temp = np.subtract(devMat[i], trainMat)*np.subtract(devMat[i], trainMat)
        dist[i] = np.sum(temp, axis = 1)
    
    resultat = lbl0[np.argmin(dist, axis = 1)]
    nberreur = sum(resultat != lbl1)
            
    # affichage du taux d'erreur
    printErr(nberreur, devMat.shape[0])
    
    return precision(nberreur*100, devMat.shape[0]), resultat

def supprVarianceInf(val=0, trainMat=X0, devMat=X1):
    # varTrainMat contient les variances d'entrainement (784,)
    varTrainMat =np.var(trainMat, axis=0)
    
    # varTrainMat contient les variances de developpement (784,)
    varDevMat =np.var(devMat, axis = 0)
    
    # tableaux des indices constants dans la matrice de developpement et d'entrainement
    trainKeep = varTrainMat>val
    devKeep = varDevMat>val
    
    # keepBoth contient les indices des luminosités qui sont constantes dans la matrice de developpement et d'entrainement
    keepBoth = devKeep[devKeep==trainKeep]
    
    # On reduit la matrice de developpement et d'entrainement pour avoir la même taille que les valeurs constantes
    trainMatKeep = trainMat[:,devKeep==trainKeep]
    trainDevKeep = devMat[:,devKeep==trainKeep]
       
    # On ne garde que les valeurs non constantes de la matrice d'entrainement
    noConstTrainMat = trainMatKeep[:,keepBoth]
    
    # On ne garde que les valeurs non constantes de la matrice de developpement
    noConstDevMat = trainDevKeep[:,keepBoth]
    
    return noConstTrainMat, noConstDevMat

def confmat(true, pred):
    dim = max(pred)+1
    z = dim*true + pred
    zcount = np.bincount(z, minlength = dim*dim)

    # precision et rappel
    zcount_reshaped = zcount.reshape(dim,dim)
    
    zcount_reshaped_sumcolonne = np.sum(zcount_reshaped, axis=1)
    zcount_reshaped_sumligne = np.sum(zcount_reshaped, axis=0)
    
    for i in range(0,zcount_reshaped_sumcolonne.shape[0]):
        # colonne (precision)
        valeur_precision = np.diag(zcount_reshaped)/float(zcount_reshaped_sumcolonne[i])  
        # ligne (rappel)
        valeur_rappel = np.diag(zcount_reshaped)/float(zcount_reshaped_sumligne[i])  
    
    print("Matrice de confusion")
    print(zcount_reshaped)

    print("Matrice de precision")
    print(valeur_precision)
    
    print("Matrice de rappel")
    print(valeur_rappel)

def main():  
    # DMIN sans ACP :
    print("DMIN normal (sans traitement)")
    print X0.shape
    print X1.shape
    res, prediction = predictMoy(X0, X1)
    confmat(lbl1, prediction)
    
    # DMIN avec ACP :
    print("DMIN avec ACP")
    # calcul de la reduction de X0 et X1 pour les obtenir en dimension p
    XRed0, XRed1, P = reduceMat(100, X0, X1)
    print XRed0.shape
    print XRed1.shape
    res, prediction = predictMoy(XRed0, XRed1)
    confmat(lbl1, prediction)
    
    # DMIN en supprimant les valeurs constantes :
    print("DMIN avec suppression des valeurs constantes")
    noConstX0, noConstX1 = supprVarianceInf(0, X0, X1)
    print noConstX0.shape
    print noConstX1.shape
    res, prediction = predictMoy(noConstX0, noConstX1)
    confmat(lbl1, prediction)
    
    
    # PPV avec ACP
    print("PPV avec ACP")
    XRed0, XRed1, P = reduceMat(100, X0, X1)
    res, prediction = predictPPV(XRed0, XRed1)
    confmat(lbl1, prediction)
    
    # PPV sans ACP
    print("PPV sans ACP")
    res, prediction = predictPPV(XRed0, XRed1)
    confmat(lbl1, prediction)
    
    # DMIN en supprimant les valeurs constantes :
    print("PPV avec suppression des valeurs constantes")
    noConstX0, noConstX1 = supprVarianceInf(0, X0, X1)
    print noConstX0.shape
    print noConstX1.shape
    res, prediction = predictPPV(noConstX0, noConstX1)
    confmat(lbl1, prediction)

    # PPV avec plusieurs ACP
    #tabPrecision = np.zeros(20)
    #for i in range(10, 210, 10):
    #    print("avec ", i, " vecteurs")
    #    tabPrecision[((i-10)/10)], res = predictPPV(X0, X1)
        
    #for i in range(500, 5000, 500):
    #    val = ((i-500)/500)+11
    #    print("avec ", i, " vecteurs")
    #    tabPrecision[val], _unused = predictPPV(X0, X1)
   
    #plt.plot(tabPrecision)
    #plt.ylabel("Taux d'erreur")
    #plt.xlabel("Taille du vecteur choisi (/10)")
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


    
    
    
    
