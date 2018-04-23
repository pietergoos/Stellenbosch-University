'''Module containing a DensityFunc abstract class, with common probability densities

@since: Jan 10, 2013

@author: kroon
'''

import numpy as np


class Gaussian(object):
    '''
    Class for representing a multi-dimensional Gaussian distribution of dimension d, 
    given mean and covariance.
    The covariance matrix has to be  positive definite and non-singular.
    
    Parameters
    ----------
    
    mean : (d,) ndarray
       mean of the distribution
    cov  : (d,d) ndarray
       Covariance matrix. 
    
    Methods
    -------
    
    f 
       Returns the value of the density function
    logf
       Returns the log of the density function
    likelihood
       Returns the likelihood of the data
    loglik
       Returns the log-likelihood of the data
    negloglik
       Returns the negative log-likelihood of the data
    sample
       Returns samples drawn from the normal distribution with the given
       mean and covariance
    get_mean
       Returns the mean
    get_cov
       Returns the covariance
    
    
    Example
    -------
    >>> from gaussian import Gaussian
    >>> # Scalar example
    >>> mean = [10.]
    >>> cov  = [[1.]]
    >>> ga   = Gaussian(mean,cov)
    >>> ga.f([10.])
        0.398942280401        
    >>> x = np.array([[10.,10.,10.]])
    >>> ga.likelihood(x)
        0.0634936359342
    >>> # Multivariate example
    >>> mean = [10.0, 10.0]
    >>> cov  = [[  1.   0.],[  0.  10.]]
    >>> ga   = Gaussian(mean,cov)
    >>> ga.f(np.array([10.,10.])
           0.050329212104487035
    >>> x = np.array([[10.,10.,10.,10.],[10.,10.,10.,10.]])
    >>> ga.likelihood(x)
           6.4162389091777101e-06
    
    '''
    def __init__(self, mean=np.array([0.,0.]), cov=np.array([[1.,0.],[0.,1.]])):
        '''
        Multivariate Gaussian class. Initiated with mean and covariance.
        
        Parameters
        ----------
        mean: (d,) ndarray
            The mean
        cov: (d,d) ndarray
            The covariance
        '''
        if (len(mean.shape)>1):
            raise ValueError('The mean is of shape', mean.shape, 'shape (d,) expected')
        if (len(cov.shape)>2):
            raise ValueError('cov is of shape',cov.shape,'shape (d,d) expected')
        if not mean.shape[0] == cov.shape[0]:
            raise ValueError('The dimensions of the mean and covariance  are not consistent')
        
        
        d = cov.shape[0]
        
        self._dim = d
        self._mean = mean.flatten()
        self._cov = cov
        self._covdet = np.linalg.det(2.*np.pi*cov)
    
        
            
    def f(self, x):
        '''
        Calculate the value of the normal distributions at x
        
        Parameters
        ----------
        x : (d,) ndarray
           Evaluate a single d-dimensional samples x
           
        Returns
        -------
        val : scalar
           The value of the normal distribution at x.
        
        '''
        
        return np.exp(self.logf(x))
    
    def logf(self, x):
        '''
        Calculate  the log-density at x
        
        Parameters
        ----------
        x : (d,) ndarray
           Evaluate the log-normal distribution at a single d-dimensional 
           sample x
           
        Returns
        -------
        logf : scalar
           The value of the log of the normal distribution at x.
        '''
        
        if (len(x.shape)>1):
            print ('x.shape = ',x.shape)
            raise ValueError('x is needs to a vector of shape (d,)')
        if (len(self._mean.shape)>1):
            print ('mean.shape = ', self._mean.shape)
            raise ValueError('the mean needs to a vector of shape (d,)')
        if (len(self._cov.shape)>2):
            print ('cov.shape = ', self._cov.shape)
            raise ValueError('cov is of shape',self._cov.shape,'shape (d,d) expected')
        if not x.shape[0] == self._mean.shape[0]:
            print ('x.shape = ',x.shape)
            print ('mean.shape = ',self._mean.shape)
            raise ValueError('The dimensions of x and the mean are not consistent')            
        if not x.shape[0] == self._cov.shape[0]:
            print ('x.shape = ',x.shape)
            print ('cov.shape = ',self._cov.shape)
            raise ValueError('The dimensions of x and the covariance matrix are not consistent')
        
        x = x[:,np.newaxis]
        mean = self._mean[:,np.newaxis]
        trans = x - mean
        
        
        mal   = -trans.T.dot(np.linalg.solve(self._cov,trans))/2.
        logf = -0.5*np.log(self._covdet) + mal
        
        return logf


    def likelihood(self, x):
        '''
        Calculates the likelihood of the data set x for the normal
        distribution.
        
        Parameters
        ----------
        x :  (d,n) ndarray
           Calculate the likelihood of n, d-dimensional samples
           
        Returns
        -------
        val : scalar
           The likelihood value   
        '''
        return np.exp(self.loglik(x))

    def loglik(self, x):
        '''
        Calculates  the log-likelihood of the dataset x for the normal 
        distribution.
        
        Parameters
        ----------
        x :  (d,n) ndarray
           Calculate the log-likelihood of n, d-dimensional samples
           
        Returns
        -------
        val : scalar
           The log-likelihood value
        '''
        return np.sum(np.apply_along_axis(self.logf, 0, x))
        
    def negloglik(self, x):
        '''
        Calculates  the negative log-likelihood of the dataset x for the 
        normal distribution.
        
        Parameters
        ----------
        x :  (d,n) ndarray
           Calculate the log-likelihood of n, d-dimensional samples
           
        Returns
        -------
        val : float
           The negative log-likelihood value
        '''
        return -self.loglik(x)
        


    def sample(self, n=1):
        '''
        Calculates n independent points sampled from the normal distribution
        
        Parameters
        ----------
        n : int
           The number of samples
           
        Returns
        -------
        samples : (d,n) ndarray
           n, d-dimensional samples
        
        '''

        return np.random.multivariate_normal(self._mean, self._cov, n).T
    
    def get_mean(self):
        """
        Returns the mean
        
        Returns
        -------
        
        mean : (d,) ndarray
            The mean
        """
        return self._mean
    
    def get_cov(self):
        """
        Returns the covariance
        
        Returns
        -------
        
        cov : (d,d) ndarray
            covariance
        """
        return self._cov

       
        

    
