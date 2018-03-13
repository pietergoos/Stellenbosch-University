import numpy as np
import scipy as sp

#################################
## THIS CLASS DOES NOT WORK!!! ##
#################################

class myNB():
    def __init__(self):
        pass

    def fit(self, X, y):
        a, b = np.unique(y, return_counts=True)
        a = np.size(y)

        Xp = np.c_[X, y]
        Xc = []
        Xs = []
        self.PCj = []
        self.leng = len(set(y))
        for i in set(y):
            Xc.append(X[y == i, :])
            Xs.append(X[y == i, :])
            self.PCj.append(b[i] / a)

        avg = []
        cov = []
        self.PCjx = []
        pxCj = []

        for j in range(0, self.leng ):
            avg.append(np.mean(a=Xc[j], axis=0))
            cov.append(np.cov(X.T))
            a = sp.stats.multivariate_normal( mean = avg[j], cov = cov[j])
            pxCj.append(a.pdf(Xs[j]))
            self.PCjx.append(pxCj[j] * self.PCj[j])

        #print(np.array(out).shape)
        # Likleyhood P(Cj|x) = P(Cj) p(x|Cj)

    def predict(self):
        pxCj = []
        for j in range(0, self.leng):
            #p(x|Cj) = P(Cj|x) / P(Cj)
            pxCj.append(self.PCjx[j] / self.PCj[j])

        #print(pxCj)
        return pxCj

        #use likelyhood formula and now want p(c|Cj) with same P(Cj) ?(RICHARD THINKS)
