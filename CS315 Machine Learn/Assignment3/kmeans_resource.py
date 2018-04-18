""" K-means algorithm. 
@since: 2017

@author: Ben        
"""
import numpy as np
class Kmeans(object):
    """ 
    Clustering using the K-means algorithm.  
    
            
    The K-means algorithm is an unsupervised classification algorithm. The number 
    of classes, k, and n, d-dimensional observations are provided. 
    This class provides the following possibilities:
    1. Partially observed data (no labels available): For initialization two
       options are available. (a) Initial cluster means are obtained from random
       samples of the data, (b) The initial means are provided. In this case the 
       number of clusters is derived from the number of initial means provided.
    2. Some of the data is fully observed, i.e. some of the data comes with labels.
       It is assumed that all the classes are represented and the initial means 
       are calculated from the labeled data. 
    
    
    
    Parameters
    ----------

    data : (n,d) ndarray 
        n, d-dimensional observations 
    codes : int, or (d,k) ndarray 
        The number of clusters to form as well as the number of
        centroids to generate. If `minit` initialization string is
        'matrix', or if an ndarray is given instead, it is
        interpreted as initial clusters to use instead.
    it : int
        Number of iterations of the k-means algrithm.
    minit : string
        Method for initialization. Available methods are 
        'points',  and 'matrix':

        'points': choose k observations (rows) at random from data for
        the initial centroids.
        
        'matrix': interpret the codes parameter as an (k,d) ndarray (or length k
            array for one-dimensional data) array of initial centroids.
    
        
            
    
        
    Methods:
    -------
    fit:
        Fit data to model, i.e. calculate the clusters
    predict:
        Assign vectors to clusters
    get_means:
        Returns the cluster centres

    Examples
    --------
    >>># Random initialization
    >>>from kmeans import Kmeans 
    >>>data = np.array([[1.9,1.5,0.4,0.4,0.1,0.2,2.0,0.3,0.1],
                          [2.3,2.5,0.2,1.8,0.1,1.8,2.5,1.5,0.3]])
    >>>codes = 3
    >>>km = Kmeans(data,codes)
    >>>print 'Class labels = ', km.label
    Class labels =  [1 1 0 2 0 2 1 2 0]
    >>>print('Due to the random initialization, different (wrong) labels
                                  are often returned')    
    >>>km.plot()  
    >>>x = np.array([0.25,2.0])
    >>>km.classify(x)
    >>>print('Verify the answer using the graph.')
    
 
    >>># Specify the initial cluster means.        
    >>>codes =  np.array([data[:,0],data[:,2],data[:,3]]).T 
    >>>km = Kmeans(data,codes)  
    >>>print 'Clusters = ',km.cluster
    Clusters =  [[ 1.8  0.2  0.3 ]
               [ 2.43333333  0.2  1.7]]
    >>>print 'Class labels = ', km.label
    Class labels =  [0 0 1 2 1 2 0 2 1]
    >>>km.plot()
   
           
    """
    def __init__(self,  codes=3, itr=10,  minit='points'):
        """ 
        Clustering using the K-means algorithm.  
    
            
        The K-means algorithm is an unsupervised classification algorithm. The number 
        of classes, k, and n, d-dimensional observations are provided. 
        This class provides the following possibilities:
        1. Partially observed data (no labels available): For initialization two
           options are available. (a) Initial cluster means are obtained from random
           samples of the data, (b) The initial means are provided. In this case the 
           number of clusters is derived from the number of initial means provided.
        2. Some of the data is fully observed, i.e. some of the data comes with labels.
           It is assumed that all the classes are represented and the initial means 
           are calculated from the labeled data. 
    
    
    
        Parameters
        ----------
        codes : int, or (d,k) ndarray 
            The number of clusters to form as well as the number of
            centroids to generate. If `minit` initialization string is
            'matrix', or if an ndarray is given instead, it is
            interpreted as initial clusters to use instead.
        itr : int
            Number of iterations of the k-means algrithm.
        minit : string
            Method for initialization. Available methods are 
            'points',  and 'matrix':

            'points': choose k observations (rows) at random from data for
            the initial centroids.
        
            'matrix': interpret the codes parameter as an (k,d) ndarray (or length k
                array for one-dimensional data) array of initial centroids.
    
        
            
        
        
        Methods:
        -------
        
        fit:
            Fit data to model, i.e. calculate the clusters
        predict:
            Assign vectors to clusters
        get_means:
            Return the cluster centres
        """
              
        
        if type(codes) is int:
            k = codes
            if not minit == 'points':
                raise Warning('Initialize using given means.')            
                
        elif not type(codes) is int:
            k,d = codes.shape
                  
        self.codes    = codes
        self.itr      = itr
        self.minit    = minit
        self.k        = k

    def fit(self, X, X_label=None,y=None):
        """ Fit the data to the model 
        Parameters
        ----------

        X : (n,d) ndarray 
            n, d-dimensional observations (no labels attached)
        X_label : (n1,d) ndarray
            n1, d-dimensional data (labels available)
        y : (n1,) array
            Provide the labels for each of the observations in X_label
            
        """
       
        codes = self.codes
        k = self.k
        itr = self.itr
        minit = self.minit
       
        n,d = X.shape
        
        if not X_label is None and not type(codes) is int:
            raise ValueError('Specify the number of clusters. Initial means calculated from labeled data')
        
        
        if X_label is None:
            means = self.init_means_(X)
        else:
            means = self.init_means_lbl_(X_label, y)
        
        for j in range(itr):
            err = X[:,np.newaxis,:]-means[:,:] 
            err2 = np.linalg.norm(err,axis=2)
            labels = np.argmin(err2,axis=1)
           
            for i in range(k):
                dat = X[labels==i,:]
                if not X_label is None:
                    dat = np.vstack([dat, X_label[y == i]])
                    
                means[i] = np.mean(dat,axis=0)        
        self.means = means   
            
        
    def predict(self,X):
        """ Assign labels to all the observations in X
         
        Parameters
        ----------

        X : (n,d) ndarray 
            n, d-dimensional observations 
        
        Returns
        -------
        labels : (n,) int array
            The cluster labels for all the observations in X
        """
        
        means = self.means
        err = X[:,np.newaxis,:]-means[:,:] 
        err2 = np.linalg.norm(err,axis=2)
        labels = np.argmin(err2,axis=1)
        return labels
        
               

    def init_means_lbl_(self,X_label,y):
        """ Calculate the initial means if labeled data is avialable
         
        Parameters
        ----------

        X_label : (n,d) ndarray 
            n, d-dimensional labeled observations 
        y : (n,) int array
            The labels of all the observations is X_label
        
        Returns
        -------
        means : (k,d) ndarray
            The initial means for each of the k clusters.
        """
        
        k = self.k
        d = X_label.shape[1]
        means = np.zeros((k,d))
        for i in range(k):
            means[i] = np.mean(X_label[y == i], axis=0)

        return means


    def init_means_(self,X):
        """ Calculate the initial means if no labeled data is avialable.
            Intialize either by random samples, or use the available means
            if specified.
         
        Parameters
        ----------

        X : (n,d) ndarray 
            n, d-dimensional labeled observations 
        
        
        Returns
        -------
        means : (k,d) ndarray
            The initial means for each of the k clusters.
        """
        
        n,d = X.shape
        codes = self.codes
        minit = self.minit 
        k = self.k
        
        # If the initial means are not given
        if type(codes) is int:            
                       
            if not minit == 'points':
                raise Warning('Use random initialization.')
           
            rnd = np.random.randint(0,n-1,size=k)
            
            rnd = np.unique(rnd)
            n_rnd = len(rnd)
            np.append(rnd,10)
                   
            s=0
            while n_rnd < k and s<500:
                rnd = np.append(rnd,np.random.randint(0,n-1))
                rnd = np.unique(rnd)
                n_rnd = len(rnd)
                s +=1
            
            means = X[rnd,:]          
        # Use the given means when available      
        elif not type(codes) is int:
            k,d1 = codes.shape
            if not d==d1:
                raise ValueError('Dimensions are inconsistent')
            means = codes
        return means
        

    @property    
    def get_means(self):
        """
        Return
        ------
        cluster : (k,d) ndarray
            The k, d-dimensional cluster means
        """
        
        return self.means

    






