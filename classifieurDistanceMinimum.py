# -*- coding: utf-8 -*-

"""Classifieur à deltaance Minimum (DMIN)"""

import numpy as np

"""Lecture des donnees"""

# X0 contient une matrice 10 000 collones x 784 lignes
# chaque representation d'un chiffre est sur une collone
X0 = np.load('trn_img.npy')

# lbl0 contient les étiquettes de chacune des images parmi les chiffres de 0 à 9
# sous forme d'un tableau mono-dimensionnel 10 000 x 1
lbl0 = np.load('trn_lbl.npy')

# X1 contient une matrice 5 000 collones x 784 lignes
X1 = np.load('dev_img.npy')

# lbl1 contient les étiquettes de chacune des images parmi les chiffres de 0 à 9
# sous forme d'un tableau mono-dimensionnel 5 000 x 1
lbl1 = np.load('dev_lbl.npy')

def dstMahalanobis(x, mu):
    epsilon = np.zeros(len(x))
    epsilon = epsilon + ((x-mu)*((x-mu).T))/len(x)
    epsilonInv =  np.linalg.inv(epsilon)
    return np.transpose(x-mu)*epsilonInv*(x-mu)


# moyenne de chaque ligne
def moyenne(x):
    moy = np.zeros((10, x.shape[1]))
    for i in range(0, 10):
        moy[i] = np.average(x[lbl0 == i,:], axis=0)
    return moy
   

# Performance : Taux erreur
def precison(nbExemplesMalClasses, nbTotalExemples):
    return nbExemplesMalClasses/nbTotalExemples  
    
def main():
	# covariance = np.cov(moy)
	# diagonale = np.diag(covariance)

	# On calcule la moyenne
	# moy est la moyenne de la distance entre les points de chacun des chiffres
	# moy est une matrice de 10 collones et 784 lignes
	moy = moyenne(X0)
	
	
	# delta doit contenir toutes les differences entre une image et la moyenne calculée
	delta = np.zeros((X1.shape[0], 10))
	# resultat contiendra le minimum des deltas et devrai donc avoir une taille de 5 000 collones
	resultat = np.zeros(X1.shape[0])

	# nberreur contiendra le nombre d'erreur total
	nberreur = 0

	for j in range(X1.shape[0]):
		# pour chaque chiffre on calcule le delta
		for i in range(0, 10):
		    delta[j][i] = np.sum(np.subtract(X1[j],moy[i])*np.subtract(X1[j],moy[i]))
                resultat[j] = np.argmin(delta[j], axis=0)
		
		# on compte les erreurs
		res1 = resultat[j]
		res2 = lbl1[j]
		if res1 != res2:
		    nberreur = nberreur + 1
		
	print("Le pourcentage d'erreur est de : ", precison(nberreur*100.0, X1.shape[0]), "%")

	# Afficher une image individuellement dans sa dimension initiale 28 x 28
	#import matplotlib.pyplot as plt
	#img = X0[0].reshape(28,28)
	#plt.imshow(img, plt.cm.gray)
	#plt.show() 
    
# Fonction main
if __name__ == "__main__":
    main()    


