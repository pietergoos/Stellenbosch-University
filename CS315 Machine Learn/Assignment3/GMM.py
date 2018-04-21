import numpy as np
import scipy as sp
from kMeans import myBetterKM as KM

###################################################
## THIS CLASS CALCULATES GAUSSIAN MIXTURE MODELS ##
##      Many thanks to gorrox23 @ GITLAB.com     ##
###################################################

class myGMM():
    def classify(self, X, k):
        #STUFF HERE
        pi, mu, Sig = self.getKMVars(X, k)

        notConverged = True
        while(notConverged):
            print("run")
            print(np.array(pi).shape)
            print(np.array(mu).shape)
            print(np.array(Sig).shape)
            #E
            gamma = self.getGamma(X, pi, mu, Sig, k)
            self.relabel(X, k, gamma)

            #M
            pi, mu, Sig = self.maximisation(X, self.y, gamma)

            #Comparison
            notConverged = not(np.array_equal(self.yold, self.y))

        return self.y, np.array(mu)


    def getKMVars(self, X, k):
        y, u = KM().classify(X, k)
        pi = []
        cov = []
        num, cn = np.unique(y, return_counts = True)
        for i in num:
            pi.append(cn[i] / y.shape[0])
            cov.append(np.cov(X[y==i].T))
        pi = np.array(pi)
        cov = np.array(cov)
        self.y = y
        return pi, u, cov

    def getGamma(self, X, pi, mu, Sig, k):
        normals = []
        gamma = []
        for j in range(k):
            normals.append(pi[j] * sp.stats.multivariate_normal(mean = mu[j], cov = Sig[j]).pdf(X))
        for j in range(k):
            gamma.append(np.array(normals)[j]/np.sum(normals, axis=0))
        return np.array(gamma)

    def relabel(self, X, k, gamma):
        self.yold = self.y
        yn = []
        for i in range(X.shape[0]):
            ptgamma = []
            for j in range(k):
                ptgamma.append(gamma[j][i])
            yn.append(np.argmax(ptgamma))
        self.y = np.array(yn)

    def maximisation(self, X, y, gamma):
        '''
        mean = []
        cov = []
        pi = []
        Nj = []
        classes, cn= np.unique(y, return_counts=True)
        for i in classes:
            Nj.append(np.sum(gamma[i]))
            pi.append(Nj[i] / y.shape[0])
            mean.append(np.sum(gamma.T[i] * X[i])/Nj[i])
            cov.append(np.sum(gamma.T[i] * (X[i] - mean[i])* (X[i] - mean[i]).T)/Nj[i])
        pi = np.array(pi)
        cov = np.array(cov)
        mean = np.array(mean)
        '''
        pi = []
        cov = []
        mean = []
        num, cn = np.unique(y, return_counts = True)
        for i in num:
            pi.append(cn[i] / y.shape[0])
            cov.append(np.cov(X[y==i].T))
            mean.append(np.mean(X[y==i], axis = 0))
        pi = np.array(pi)
        cov = np.array(cov)
        mean = np.array(mean)


        return pi, mean, cov
