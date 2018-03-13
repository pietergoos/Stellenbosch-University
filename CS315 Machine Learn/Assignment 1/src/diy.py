import numpy as np
import scipy as sp

class myPCA():
    def __init__(self, num_comp = 1, whiten=False, SVD = False):
        self.num_comp = num_comp
        self.whiten = whiten
        self.useSVD = SVD

    def fit(self, data):
        n, m = np.shape(data)                                                       #row x col of data
        avg = np.mean(a=data, axis=0)                                               #gets verical mean of data ((X,Y), (X,Y))
        avg_l = len(avg)
        avg = avg.reshape(avg_l, 1)                                                 #Rotates avg to be able to subtract
        self.data_cent = data.T - avg                                               #Centered Data

        if(self.useSVD == False):
            self.Cov = np.cov(self.data_cent)
            self.bigLambda, self.U = np.linalg.eig(self.Cov)                        #U - principal dir
        else:
            self.U, self.S, self.vh = np.linalg.svd(self.data_cent)                 #S - Sigma // vh - whitened output // u - eigenvector mat
            self.bigLambda = (1/(n)) * (self.S ** 2)

        self.evr = self.bigLambda[:self.num_comp]/np.sum(self.bigLambda)            #Explained Variance Ratio

    def transform(self, data):
        trnsfm = self.U[:, :self.num_comp]
        newLambda = np.diag(self.bigLambda)[:self.num_comp, :self.num_comp]
        if(self.whiten == False):
            out = (trnsfm.T.dot(self.data_cent)).T
        else:
            nl = sp.linalg.fractional_matrix_power(newLambda, -0.5)
            one = trnsfm.dot(nl).T
            out = one.dot(self.data_cent).T
        return out
