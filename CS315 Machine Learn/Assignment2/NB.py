import numpy as np
import scipy as sp

###############################################################
## THIS CLASS CALCULATES THE GAUSSIAN NAIVE BAYES PREDICTION ##
###############################################################

class myNB():
    def __init__(self):
        pass

    def fit(self, X, y):
        Xc = []
        self.PCj = []
        self.avg = []
        self.cov = []
        std = []

        #Save class names
        self.Cnames, self.Cnumj = np.unique(y, return_counts=True) #Gets the names of the classes and how many data points belong to each
        self.numClass = np.size(self.Cnames) #Gets the number of classes
        self.totalPts = np.size(y) #Gets the total number of Data Points in the training data set

        for i in self.Cnames:
            Xc.append(X[y==i, :]) #Gets data from training set and splits it according to class

        for i in range(0, self.numClass):
            self.PCj.append(self.Cnumj[i] / self.totalPts) #Calculates P(Cj) for this class
            self.avg.append(np.mean(a=Xc[i], axis = 0)) #Calculates the average per class
            std.append(np.diag(np.std(a=Xc[i], axis = 0))) #Places standard deviation along the diagonal of a matrix
            #self.cov.append(np.cov(Xc[i].T)) #Calculates the per class covariance
            self.cov.append(std[i]) #Appends the correct standard deviation along the diagonal of a matrix to the cov list

    def predict(self, X):
        out = []
        for j in range(0, np.shape(X)[0]):  #Steps through number of points given
            PxCj = [] #Creates empy array per point for P(x|Cj)
            for i in range(0, self.numClass): #Steps through classes
                a = sp.stats.multivariate_normal(mean = self.avg[i], cov = self.cov[i])
                PxCj.append(a.pdf(X[j])*self.PCj[i]) #Calculates Gaussian P(x|Cj) * P(Cj)
            maxV = np.argmax(PxCj) #Returns index of class
            out.append(self.Cnames[maxV]) #Gets name of class at max index
        return np.array(out) #returns list of classes to which the datapoints belong
