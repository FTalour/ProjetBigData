# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:33:31 2016

@author: Tristan Le Nair & Florian Talour
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

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

def predictMoySimple():
    
    # entrainement
    moy = moyenne(X0)

    # definition des tableaux
    dist = np.zeros((10, X1.shape[0]))
    resultat = np.zeros(X1.shape[0], dtype = np.int)

    nberreur = 0
    # calcul du taux d'erreur
    for i in range(0, 10):
        temp = np.subtract(X1,moy[i])*np.subtract(X1,moy[i])
        dist[i] = np.sum(temp, axis=1)
            
    resultat = np.argmin(dist, axis = 0)
    nberreur = sum(resultat != lbl1)

    # affichage du taux d'erreur
    printErr(nberreur, X1.shape[0])
    return precision(nberreur*100, X1.shape[0]), resultat


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
def predictPPVSimple():
    
    # definition des tableaux
    dist = np.zeros((X1.shape[0], X0.shape[0]))
    resultat = np.zeros(X1.shape[0], dtype = np.int)
    
    nberreur = 0
    
    # calcul du taux d'erreur
    for i in range(X1.shape[0]):
        temp = np.subtract(X1[i], X0)*np.subtract(X1[i], X0)
        dist[i] = np.sum(temp, axis = 1)
    
            
    resultat = lbl0[np.argmin(dist, axis = 1)]
    nberreur = sum(resultat != lbl1)
            
    # affichage du taux d'erreur
    printErr(nberreur, X1.shape[0])
    
    return precision(nberreur*100, X1.shape[0]), resultat

def predictPPV(p):
   	
    # calcul de la reduction de X0 et X1 pour les obtenir en dimension p
    XRed0, XRed1, P = reduceMat(p)
    
    # soit un sous-ensemble de la base d'apprentissage
    #XRed0 = trainPPV(XRed0,p) # on enlève p valeurs à X0
    
    # definition des tableaux
    dist = np.zeros((X1.shape[0], X0.shape[0]))
    resultat = np.zeros(X1.shape[0], dtype = np.int)
    
    nberreur = 0
    
    # calcul du taux d'erreur
    for i in range(XRed1.shape[0]):
        temp = np.subtract(XRed1[i], XRed0)*np.subtract(XRed1[i], XRed0)
        dist[i] = np.sum(temp, axis = 1)
    
            
    resultat = lbl0[np.argmin(dist, axis = 1)]
    nberreur = sum(resultat != lbl1)
            
    # affichage du taux d'erreur
    printErr(nberreur, XRed1.shape[0])
    
    return precision(nberreur*100, XRed1.shape[0]), resultat

def confmat(true, pred):
    dim = max(pred)+1
    z = dim*true + pred
    zcount = np.bincount(z, minlength = dim*dim)

    # precision et rappel
    zcount_reshaped = zcount.reshape(dim,dim)
    
    zcount_reshaped_sumcolonne = np.sum(zcount_reshaped, axis=1)
    zcount_reshaped_sumligne = np.sum(zcount_reshaped, axis=0)
    
    for i in range(0,zcount_reshaped_sumcolonne.shape[0]):
        valeur_precision = np.diag(zcount_reshaped)/float(zcount_reshaped_sumcolonne[i])  # colonne (precision)
        valeur_rappel = np.diag(zcount_reshaped)/float(zcount_reshaped_sumligne[i])  # ligne (rappel)
    
    print("Matrice de confusion")
    print(zcount_reshaped)

    print("Matrice de precision")
    print(valeur_precision)
    
    print("Matrice de rappel")
    print(valeur_rappel)

def main():  
    # DMIN sans ACP :
    #t1 = np.datetime64(dt.datetime.now())
    #res, prediction = predictMoySimple()
    #t2 = np.datetime64(dt.datetime.now())
    #print ("Temps DMIN sans ACP : ", t2 - t1)
    #confmat(lbl1, prediction)
    
    # DMIN avec plusieurs ACP
    #tabPrecision = np.zeros(20)
    #for i in range(10, 210, 10):
    #    print("avec ", i, " vecteurs")
    #    t5 = np.datetime64(dt.datetime.now())
    #    tabPrecision[((i-10)/10)], res = predictMoy(i)
    #    t6 = np.datetime64(dt.datetime.now())
    #    print ("Temps : ", t6 - t5)

    # 1PPV sans ACP
    t3 = np.datetime64(dt.datetime.now())
    res, prediction = predictPPVSimple()
    t4 = np.datetime64(dt.datetime.now())
    print ("Temps 1PPV sans ACP : ", t4 - t3)
    confmat(lbl1, prediction)

    # 1PPV avec plusieurs ACP
    #tabPrecision = np.zeros(20)
    #for i in range(10, 210, 10):
    #    print("avec ", i, " vecteurs")
    #    t7 = np.datetime64(dt.datetime.now())
    #    tabPrecision[((i-10)/10)], res = predictPPV(i)
    #    t8 = np.datetime64(dt.datetime.now())
    #    print ("Temps : ", t8 - t7)
    
        
    #for i in range(500, 5000, 500):
    #    val = ((i-500)/500)+11
    #    print("avec ", i, " vecteurs")
    #    tabPrecision[val], _unused = predictPPV(i)
   
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


    
    
    
    
