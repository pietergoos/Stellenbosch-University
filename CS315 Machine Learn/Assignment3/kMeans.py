import numpy as np
import scipy as sp

##############################################
## THIS CLASS CALCULATES K-MEANS CLUSTERING ##
##############################################

#TODO Implement whitening / normalizing
#TODO Implement binary split as an option

class myKM():
    def __init__(self):
        pass

    def train(self, X, nClasses, norm = False, bSplit = False, dim = 2):
        classes = []
        finished = False

        for i in range(nClasses):
            classes.append(i)

        self.classes = classes
        rMeans = np.random.rand(nClasses, dim) #each label gets components for each dimension

        while not finished:
            uN = self.refresh(X, rMeans, classes)
            finished = self.isClose(rMeans, uN, 0.0)
            rMeans = uN

    def isClose(self, wOld, wNew, threshold):
        return np.any(np.abs(np.array(wOld) - np.array(wNew))) <= threshold

    def refresh(self, X, uOld, classes):
        y = []
        Xo = []
        for i in range(len(X)):
            dis = []
            for j in range(len(classes)): #This should be the number of labels
                dis.append(np.linalg.norm(X[i] - uOld[j]))
            maxi = np.argmin(np.array(dis))
            y.append(classes[maxi])
        y = np.array(y)
        u = []
        for i in range(len(classes)):
            Xo.append(X[y==i, :])
            u.append(np.mean(a=Xo[i], axis = 0))
        u= np.array(u)
        self.y = y
        self.u = u
        return u
