import numpy as np
import scipy as sp

##############################################
## THIS CLASS CALCULATES K-MEANS CLUSTERING ##
##   Many thanks to gorrox23 @ GITLAB.com   ##
##############################################

class myKM():
    def __init__(self):
        pass

    def train(self, X, nClasses, dim = 2):
        classes = [0,1]
        #finished = False
        iter = 2
        self.eig = []

        self.classes = classes
        rMean = []
        for i in range(2):
            rMean.append(X[int(np.random.rand(1)[0]*X.size/dim)])
        rMeans = np.array(rMean)

        self.solver(X, rMeans, classes)
        maxL = self.bigVar(classes)
        #oldX = self.X
        oldXo = self.Xo
        oldy = self.y
        while(iter < nClasses):
            cN = [classes[maxL], len(classes)]
            rM = []
            for i in range(2):
                rM.append(self.Xo[maxL][int(np.random.rand(1)[0]*np.array(self.Xo).size/dim)])
            self.solver(self.Xo[maxL], np.array(rM), cN)
            classes.append(len(classes))
            maxL = self.bigVar(classes)
            iter += 1
        self.y = oldy
        #classes, y, u

    def isClose(self, wOld, wNew, threshold):
        return np.any(np.abs(np.array(wOld) - np.array(wNew))) <= threshold

    def solver(self, X, rMeans, classes):
        finished = False
        while not finished:
            uN = self.refresh(X, rMeans, classes)
            finished = self.isClose(rMeans, uN, 0.0)
            rMeans = uN

    def bigVar(self, classes):
        self.eig = []
        lenn = []
        for i in range(len(classes)):
            self.eig.append(np.linalg.eigvals(np.cov(self.Xo[i].T, bias=True)))
            lenn.append(self.eig[i][0]**2 + self.eig[i][1]**2)
        self.eig = np.array(self.eig)
        lenn = np.array(lenn)
        max = np.argmax(lenn)
        return max

    def refresh(self, X, uOld, classes):
        #Classes are the labels
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
        self.Xo = Xo
        self.y = y
        self.u = u
        return u

class myBetterKM():
    def classify(self, X, n_classes):
        y = np.zeros(X.shape[0], dtype=int)
        mean = self.getClassVars(X, y, "mean")
        cntr = 0

        for i in range(n_classes - 1):
            cVar = self.getClassVars(X, y, "var")
            icVar = np.argmax(cVar)
            mean.remove(mean[icVar])
            bS = self.binSplit(X[y == icVar])
            mean.append(bS[0])
            mean.append(bS[1])
            y, mean = self.basicKM(X, mean)
            mean = mean.tolist()
        return y, np.array(mean)

    def basicKM(self, X, mean):
        oldy = np.zeros(X.shape[0], dtype=int)
        notConverged = True
        while notConverged:
            newy = []
            for xn in X:
                newy.append(self.closestClass(xn, mean))
            mean = self.getClassVars(X, newy, "mean")
            notConverged = not(np.array_equal(oldy, newy))
            oldy = newy
        return np.array(newy), np.array(mean)

    def closestClass(self, xn, mean):
        dist = []
        for i in range(len(mean)):
            dist.append(np.linalg.norm(xn - mean[i]))
        return np.argmin(dist)

    def getClassVars(self, X, y, val = "mean"):
        var = []
        clss = np.unique(y)
        for i in clss:
            if(val == "mean"):
                var.append(np.mean(X[y==i], axis=0))
            if(val == "var"):
                var.append(np.linalg.norm(np.std(X[y==i], axis=0)**2))
        return var

    def binSplit(self, X):
        pts = []
        for i in range(2):
            pts.append(X[int(np.random.rand(1)[0]*X.shape[0])])
        return pts
