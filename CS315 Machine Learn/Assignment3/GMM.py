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
            #E
            gamma = self.getGamma(X, pi, mu, Sig, k)
            yLocal, maxgamma = self.relabel(X, k, gamma)

            #M
            pi, mu, Sig = self.maximisation(X, self.y, gamma)

            #Comparison
            notConverged = not(np.array_equal(self.yold, self.y))

        return self.y, np.array(mu), gamma, maxgamma, Sig

    def getKMVars(self, X, k):
        y, u = KM().classify(X, k)
        pi = []
        cov = []
        num, cn = np.unique(y, return_counts = True)
        for i in num:
            pi.append(cn[i] / y.shape[0])
            cov.append(np.cov(X[y==i].T, bias=True))
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
        maxgamma = []
        yn = []
        for i in range(X.shape[0]):
            ptgamma = []
            for j in range(k):
                ptgamma.append(gamma[j][i])
            yn.append(np.argmax(ptgamma))
            maxgamma.append(ptgamma[np.argmax(ptgamma)])
        self.y = np.array(yn)
        return np.array(yn), np.array(maxgamma)

    def maximisation(self, X, y, gamma):
        mean = []
        cov = []
        pi = []
        Nj = []
        classes, cn= np.unique(y, return_counts=True)
        for i in classes:
            Nj.append(np.sum(gamma[i]))
            pi.append(Nj[i] / y.shape[0])
            s = 0
            for n in range(X.shape[0]):
                s += gamma[i].T[n] * X[n].T
            mean.append(s/Nj[i])
            s = 0
            for n in range(X.shape[0]):
                s += gamma[i].T[n] * (X[n].T - mean[i]) * (X[n].T - mean[i]).T
            cov.append(s/Nj[i])
        pi = np.array(pi)
        cov = np.array(cov)
        mean = np.array(mean)
        return pi, mean, cov
