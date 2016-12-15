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
    
# moyenne de chaque ligne avec un label
def moyenne_X_lbl(x=X0, lbl=lbl0):
    moy = np.zeros((10, x.shape[1]))
    for i in range(0, 10):
        moy[i] = np.average(x[lbl == i,:], axis=0)
    return moy
    
# standart deviation de chaque ligne sans label
def variance_X(x=X0):
    variance = np.zeros(x.shape[1])
    variance = np.var(x, axis=0)
    return variance
    
# standart deviation de chaque ligne avec un label
def variance_X_lbl(x=X0, lbl=lbl0):
    var = np.zeros((10, x.shape[1]))
    for i in range(0, 10):
        var[i] = np.var(x[lbl == i,:], axis=0)
    return var
    
    

# 1PPV obtenir un sous-ensemble de la base d'apprentissage, on enlève p valeurs à x
def trainPPV(x, p):
    return x[:,:p] 

# ACP
def reduceMat(p=100, trainMat=X0, devMat=X1):

    C = np.cov(trainMat, rowvar=0)
    v,w = np.linalg.eigh(C)

    P = -w[:, len(w)-p : len(w)] 
    
    #trainMat_prime = np.multiply(trainMat, v)
    #devMat_prime = np.multiply(devMat, v)
    
    XRed0 = np.matmul(trainMat, P)
    XRed1 = np.matmul(devMat, P)

    return XRed0, XRed1, P

# prediction de classes par distance minimum
def predictDMIN(trainMat=X0, devMat=X1):

    # entrainement
    moy = moyenne_X_lbl(trainMat)

    # definition des tableaux
    dist = np.zeros((10, devMat.shape[0]))
    resultat = np.zeros(devMat.shape[0], dtype = np.int)

    nberreur = 0
    # calcul du taux d'erreur
    for i in range(0, 10):
        squareDst = np.subtract(devMat,moy[i])*np.subtract(devMat,moy[i])
        dist[i] = np.sum(squareDst, axis=1)
            
    resultat = np.argmin(dist, axis = 0)
    nberreur = sum(resultat != lbl1)

    # affichage du taux d'erreur
    printErr(nberreur, devMat.shape[0])
    return precision(nberreur*100, devMat.shape[0]), resultat

# prediction de classes par distance minimum
def predictDMINMahalanobisDiagonale(trainMat=X0, devMat=X1):

    # entrainement
    moy = moyenne_X_lbl(trainMat)

    # definition des tableaux
    dist = np.zeros((10, devMat.shape[0]))
    resultat = np.zeros(devMat.shape[0], dtype = np.int)

    # calcul de l'ecart type de chaque nombre
    variance = variance_X(trainMat)
    
    nberreur = 0
    # calcul du taux d'erreur
    for i in range(0, 10):
        squareDst = np.subtract(devMat,moy[i])*np.subtract(devMat,moy[i])/variance
        dist[i] = np.sum(squareDst, axis=1)

    resultat = np.argmin(dist, axis = 0)
    nberreur = sum(resultat != lbl1)

    # affichage du taux d'erreur
    printErr(nberreur, devMat.shape[0])
    return precision(nberreur*100, devMat.shape[0]), resultat

# prediction de classes par le plus proche voisin
def predict1PPV(trainMat=X0, devMat=X1):
    
    # definition des tableaux
    dist = np.zeros((devMat.shape[0], trainMat.shape[0]))
    resultat = np.zeros(devMat.shape[0], dtype = np.int)
    
    nberreur = 0
    # calcul du taux d'erreur
    for i in range(devMat.shape[0]):
        dst = np.subtract(devMat[i], trainMat)*np.subtract(devMat[i], trainMat)
        dist[i] = np.sum(dst, axis = 1)
    
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
    
def supprStdInf(val=0, trainMat=X0, devMat=X1):
    # varTrainMat contient les variances d'entrainement (784,)
    varTrainMat =np.std(trainMat, axis=0)
    
    # varTrainMat contient les variances de developpement (784,)
    varDevMat =np.std(devMat, axis = 0)
    
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
        valeur_precision = zcount_reshaped/float(zcount_reshaped_sumcolonne[i])  
        # ligne (rappel)
        valeur_rappel = zcount_reshaped/float(zcount_reshaped_sumligne[i])  
    
    print("Matrice de confusion")
    print(zcount_reshaped)

    # on arrondi les valeurs à l'affichage pour une meilleure lisibilitée
    print("Matrice de precision")
    print(np.around(valeur_precision,3)) 
    
    print("Matrice de rappel")
    print(np.around(valeur_rappel,3))

# décommenter les passages souhaités pour les executer
def main():
    '''
    # naive_bayes avec ACP
    from sklearn.naive_bayes import GaussianNB
    
    #for i in range(5, 125, 10):
    #    print i
    t1 = np.datetime64(dt.datetime.now())
    gnb = GaussianNB()
    XRed0, XRed1, P = reduceMat(45, X0, X1)
    print XRed0.shape
    print XRed1.shape
    y_pred = gnb.fit(XRed0, lbl0).predict(XRed1)
    t2 = np.datetime64(dt.datetime.now())
    print("Number of mislabeled points out of a total %d points : %d"
            % (X1.shape[0],(lbl1 != y_pred).sum()))
    printErr((lbl1 != y_pred).sum(), X1.shape[0])
    print ("Temps Naive Bayes avec ACP : ", t2 - t1)
    confmat(lbl1, y_pred)
    '''

    '''
    # 1PPV (KNeighborsClassifier) sans ACP
    from sklearn import neighbors
    t1 = np.datetime64(dt.datetime.now())
    n_neighbors = 15
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    X = X0
    y = lbl0
    Z = clf.fit(X, y).predict(X)
    t2 = np.datetime64(dt.datetime.now())
    print("Number of mislabeled points out of a total %d points : %d"
            % (X.shape[0],(y != Z).sum()))
    printErr((y != Z).sum(), X.shape[0])
    print ("Temps Mahalanobis diagonale sans ACP : ", t2 - t1)
    confmat(y, Z)
    
    # 1PPV (KNeighborsClassifier) avec ACP
    t1 = np.datetime64(dt.datetime.now())
    XRed0, XRed1, P = reduceMat(100, X0, X1)
    n_neighbors = 15
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    X = XRed0
    y = lbl0
    Z = clf.fit(X, y).predict(X)
    t2 = np.datetime64(dt.datetime.now())
    print("Number of mislabeled points out of a total %d points : %d"
            % (X.shape[0],(y != Z).sum()))
    printErr((y != Z).sum(), X.shape[0])
    print ("Temps Mahalanobis diagonale avec ACP : ", t2 - t1)
    confmat(y, Z)
    '''
    
    '''
    # DMIN normal (sans traitement)
    print("DMIN sans ACP (sans traitement)")
    print X0.shape
    print X1.shape
    t1 = np.datetime64(dt.datetime.now())
    precision, prediction = predictDMIN(X0, X1)
    t2 = np.datetime64(dt.datetime.now())
    print ("Temps DMIN sans ACP : ", t2 - t1)
    confmat(lbl1, prediction)
    
    # DMIN avec ACP
    print("DMIN avec ACP")
    t3 = np.datetime64(dt.datetime.now())
    XRed0, XRed1, P = reduceMat(90, X0, X1)
    print XRed0.shape
    print XRed1.shape
    precision, prediction = predictDMIN(XRed0, XRed1)
    t4 = np.datetime64(dt.datetime.now())
    print ("Temps : ", t4 - t3)
    confmat(lbl1, prediction)
    
    # DMIN en supprimant les valeurs constantes
    print("DMIN avec suppression des valeurs constantes")
    t5 = np.datetime64(dt.datetime.now())
    noConstX0, noConstX1 = supprVarianceInf(0, X0, X1)
    print noConstX0.shape
    print noConstX1.shape
    precision, prediction = predictDMIN(noConstX0, noConstX1)
    t6 = np.datetime64(dt.datetime.now())
    print ("Temps 1PPV en supprimant les valeurs constantes : ", t6 - t5)
    confmat(lbl1, prediction)
    '''
    
    '''
    # DMIN MahalanobisDiagonale en supprimant les valeurs constantes
    print("DMINMahalanobisDiagonale avec suppression des valeurs constantes")
    t5 = np.datetime64(dt.datetime.now())
    noConstX0, noConstX1 = supprStdInf(0, X0, X1)
    print noConstX0.shape
    print noConstX1.shape
    precision, prediction = predictDMINMahalanobisDiagonale(noConstX0, noConstX1)
    t6 = np.datetime64(dt.datetime.now())
    print ("Temps DMINMahalanobisDiagonale en supprimant les valeurs constantes : ", t6 - t5)
    confmat(lbl1, prediction)
    '''
    
    '''
    # DMIN MahalanobisDiagonale avec ACP
    print("DMINMahalanobisDiagonale avec ACP")
    t5 = np.datetime64(dt.datetime.now())
    XRed0, XRed1, P = reduceMat(90, X0, X1)
    print XRed0.shape
    print XRed1.shape
    precision, prediction = predictDMINMahalanobisDiagonale(XRed0, XRed1)
    t6 = np.datetime64(dt.datetime.now())
    print ("Temps DMINMahalanobisDiagonale avec ACP : ", t6 - t5)
    confmat(lbl1, prediction)
    '''
    
    '''
    # 1PPV avec ACP
    print("PPV avec ACP")
    t1 = np.datetime64(dt.datetime.now())
    XRed0, XRed1, P = reduceMat(100, X0, X1)
    print XRed0.shape
    print XRed1.shape
    precision, prediction = predict1PPV(XRed0, XRed1)
    t2 = np.datetime64(dt.datetime.now())
    print ("Temps 1PPV avec ACP : ", t2 - t1)
    confmat(lbl1, prediction)
    
    # 1PPV sans ACP (sans traitement)
    print("PPV sans ACP")
    t3 = np.datetime64(dt.datetime.now())
    precision, prediction = predict1PPV()
    t4 = np.datetime64(dt.datetime.now())
    print ("Temps 1PPV sans ACP : ", t4 - t3)
    confmat(lbl1, prediction)
    
    # DMIN en supprimant les valeurs constantes :
    print("PPV avec suppression des valeurs constantes")
    t5 = np.datetime64(dt.datetime.now())
    noConstX0, noConstX1 = supprVarianceInf(0, X0, X1)
    print noConstX0.shape
    print noConstX1.shape
    precision, prediction = predict1PPV(noConstX0, noConstX1)
    t6 = np.datetime64(dt.datetime.now())
    print ("Temps 1PPV en supprimant les valeurs constantes : ", t6 - t5)
    confmat(lbl1, prediction)
    '''

    '''
    # 1PPV avec plusieurs ACP et affichage dans un graphique
    tabPrecision = np.zeros(20)
    for i in range(10, 210, 10):
        print("avec ", i, " vecteurs")
        t7 = np.datetime64(dt.datetime.now())
        tabPrecision[((i-10)/10)], res = predict1PPV(X0, X1)
        t8 = np.datetime64(dt.datetime.now())
        print ("Temps : ", t8 - t7)

        
    for i in range(500, 5000, 500):
        val = ((i-500)/500)+11
        print("avec ", i, " vecteurs")
        tabPrecision[val], _unused = predict1PPV(X0, X1)
   
    plt.plot(tabPrecision)
    plt.ylabel("Taux d'erreur")
    plt.xlabel("Taille du vecteur choisi (/10)")
    plt.show()
    '''

    '''
    # Sauvegarde du meilleur pourcentage
    np.save('test-1nn', res)
    '''

    '''
    # Afficher une image individuellement
    img = XRed0[0].reshape(p/10,10)
    plt.imshow(img, plt.cm.gray)
    plt.show()
    '''

    '''  
    # Afficher une image individuellement dans sa dimension initiale 28 x 28
    img = X0[0].reshape(28,28)
    plt.imshow(img, plt.cm.gray)
    plt.show()
    '''

# fonction main
if __name__ == "__main__":
    main()   


    
    
    
    
