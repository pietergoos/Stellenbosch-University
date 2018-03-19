import numpy as np
import scipy as sp

##############################################################
## THIS CLASS CALCULATES THE LOGISTIC REGRESSION PREDICTION ##
##############################################################
## This file is largely based on LR_b.py in the same folder ##
##############################################################

class myLR():
    def __init__(self):
        pass

    def fit(self, X, y, learnRate, threshold = 0.00001):
        Xne = np.insert(X, 0, 1, axis=1)
        w = [0, 0, 0]
        #w = np.array(np.zeros([1,3]))
        finished = False
        while not finished:
            wN = self.gradDes(Xne, y, w, learnRate)
            finished = self.isClose(w, wN, threshold)
            w = wN
        self.weights = w

    def predict(self, X):
        Xne = np.insert(X, 0, 1, axis=1)
        a = []
        for i in range(Xne.shape[0]):
            a.append(np.round(self.sigmaFunc(self.weights, Xne[i])))
        return np.array(a)

    def isClose(self, wOld, wNew, threshold):
        return np.any(np.abs(np.array(wOld) - np.array(wNew))) < threshold

    def sigmoid(self, a):
        return (1/(1+np.exp(-a)))

    def hessian(self, X, W, Regul=2):
        X = np.insert(X, 0, 1, axis=1)
        hess = 0
        for i in range(len(X)):
            s = self.sigmoid(np.dot(X[i], W))
            hess += (s * (1-s)) * X[i, :].reshape((3,1)) * X[i, :].reshape((3,1)).T
        hess += (1/Regul) * np.identity(len(X[0]))
        return hess

#    This is designed to run the σ(w^T x_n)
#    w -     Weighting
#    x -     datapoint
    def sigmaFunc(self, w, x):
        z = 0

        z = np.array(w).dot(x)
        #for i in range(len(w)):
        #    z += x[i] * w[i]

        return self.sigmoid(z)

#    This is the Loss function
#    X -         dataset
#    y -         labels
#    w -         weights
#    iter -      iteration coefficient (used for derivative)
#    learnRate - learning rate of the function (1/c) multiplied in later
#    dt -        use the derivative or not - default not
    def E(self, X, y, w, it, learnRate, dt = False):
        sumLoss = 0
        for i in range(len(y)):
            xi = X[i]
            xij = xi[it]
            hyp = self.sigmaFunc(w, xi)
            if dt:
                loss = (hyp - y[i])*xij
            else:
                if y[i] == 1:
                    loss = y[i] * math.log(hyp)
                elif y[i] == 0:
                    loss = (1-y[i]) * math.log(1-hyp)
            sumLoss += loss
        return (1/learnRate) * sumLoss

#    Gradient Descent Function - used to obtain a more correct weight
#    X -         dataset
#    y -         labels
#    w -         weights from before
#    learnRate - Learning rate variable - (1/c) multiplied later
    def gradDes(self, X, y, w, learnRate):
        wNew = []
        for i in range(len(w)):
            CFD = self.E(X, y, w, i, learnRate, True)
            wNew.append(w[i] - CFD)
            #print(wNew)
        return wNew
