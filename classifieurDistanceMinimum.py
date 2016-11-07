# -*- coding: utf-8 -*-
"""Classifieur à distance Minimum (DMIN)"""

"""Lecture des donnees"""

import numpy as np

#X0 contient une matrice 10 000 x 784
X0 = np.load('data\\trn_img.npy')

#lbl0 contient les étiquettes de chacune des images parmi les chiffresde 0 à 9
#sous forme d'un tableau mono-dimensionnel 10 000 x 1
lbl0 = np.load('data\\trn_lbl.npy')


#moyenne de chaque ligne
moy = np.zeros(784)
for i in range(0, 9):
    moy = np.average(X0[lbl0 == i], axis=0)
    
#covariance = np.cov(moy)
#diagonale = np.diag(covariance)

#def train():
    
#def guess():
    

def dstMahalanobis(x, mu):
	epsilon = ((x[i]-mu)*((x[i]-mu).T))/np.ndarry.size(x)
	epsilonInv =  np.linalg.inv(epsilon)
	return np.transpose(x-mu)*epsilonInv*(x-mu)

	

#Sort points along the x-coordinate
#Split the set of points into two equal-sized subsets by a vertical line x = xmid
#Solve the problem recursively in the left and right subsets. This will give the left-side and right-side minimal distances dLmin and dRmin respectively.
#Find the minimal distance dLRmin among the pair of points in which one point lies on the left of the dividing vertical and the second point lies to the right.
#The final answer is the minimum among dLmin, dRmin, and dLRmin."""

#http://www.cs.mcgill.ca/~cs251/ClosestPair/ClosestPairPS.html

#Afficher une image individuellement dans sa dimension initiale 28 x 28
import matplotlib.pyplot as plt
img = X0[0].reshape(28,28)
plt.imshow(img, plt.cm.gray)
plt.show()

#Performance : Taux erreur
def precison(nbExemplesMalClasses, nbTotalExemples):
    return nbExemplesMalClasses/nbTotalExemples