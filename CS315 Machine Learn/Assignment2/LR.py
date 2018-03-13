import numpy as np
import scipy as sp

##############################################################
## THIS CLASS CALCULATES THE LOGISTIC REGRESSION PREDICTION ##
##############################################################

class myLR():
    def __init__(self):
        pass

    def fit(self, X, y):
        #Determine the size of y
        #Get default vals
        #repeat while error is >0.00001
            #theta2 = graddesc(X, y, theta1, length, alpha)
            #t1 = t2
            #for iterator % 100 =0:
                #Costfunct(X, y, theta1, length)

    def predict(self):
        pass

    def gradDesc(self, X, y, theta, length, alpha):
        newT = []
        c = alpha / length
        for i in range(len(theta)):
            CFD = DcostFunc(X, y, theta, i, length, alpha)
            newT.append(theta[i] - CFD)
        return newT

    def costFunc(self, X, y, theta, length):
        sigErr = 0
        for i in range(length):
            hyp = hypoth(theta, X[i])
            if y[i] == 1:
                err = y[i] * math.log(hyp)
            elif y[i] == 0:
                err = (1-y[i] * math.log(1-hyp))
            sigErr += err
        J = (-1/length) * sigErr
        return J

    def DcostFunc(X, y, theta )

    def hypoth(self, theta, X):
        z = 0
        for i in range(len(theta)):
            z += x[i]*theta[i]
        Gz = float(1.0 / float((1.0 + math.exp(-1.0*z))))
        return Gz
