'''
Module implementing Hidden Markov model parameter estimation.

To avoid repeated warnings of the form "Warning: divide by zero encountered in log",
it is recommended that you use the command "np.seterr(divide="ignore")" before
invoking methods in this module.  This warning arises from the code using the
fact that python sets log 0 to "-inf", to keep the code simple.

Initial version created on Mar 28, 2012

@author: kroon, herbst
'''

from __future__ import division
from warnings import warn
import numpy as np
from gaussian import Gaussian
np.seterr(divide="ignore")

class HMM(object):
    '''
    Class for representing and using hidden Markov models.
    Currently, this class only supports left-to-right topologies and Gaussian
    emission densities.

    The HMM is defined for n_states emitting states (i.e. states with
    observational pdf's attached), and an initial and final non-emitting state (with no
    pdf's attached). The emitting states always use indices 0 to (n_states-1) in the code.
    Indices -1 and n_states are used for the non-emitting states (-1 for the initial and
    n_state for the terminal non-emitting state). Note that the number of emitting states
    may change due to unused states being removed from the model during model inference.

    To use this class, first initialize the class, then either use load() to initialize the
    transition table and emission densities, or fit() to initialize these by fitting to
    provided data.  Once the model has been fitted, one can use viterbi() for inferring
    hidden state sequences, forward() to compute the likelihood of signals, score() to
    calculate likelihoods for observation-state pairs, and sample()
    to generate samples from the model.

    Attributes:
    -----------
    data : (d,n_obs) ndarray
        An array of the trainining data, consisting of several different
        sequences.  Thus: Each observation has d features, and there are a total of n_obs
        observation.   An alternative view of this data is in the attribute signals.

    diagcov: boolean
        Indicates whether the Gaussians emission densities returned by training
        should have diagonal covariance matrices or not.
        diagcov = True, estimates diagonal covariance matrix
        diagcov = False, estimates full covariance matrix

    dists: (n_states,) list
        A list of Gaussian objects defining the emitting pdf's, one object for each
        emitting state.

    maxiters: int
        Maximum number of iterations used in Viterbi re-estimation.
        A warning is issued if 'maxiters' is exceeded.

    rtol: float
        Error tolerance for Viterbi re-estimation.
        Threshold of estimated relative error in log-likelihood (LL).

    signals : ((d, n_obs_i),) list
        List of the different observation sequences used to train the HMM.
        'd' is the dimension of each observation.
        'n_obs_i' is the number of observations in the i-th sequence.
        An alternative view of thise data is in the attribute data.

    trans : (n_states+1,n_states+1) ndarray
        The left-to-right transition probability table.  The rightmost column contains probability
        of transitioning to final state, and the last row the initial state's
        transition probabilities.   Note that all the rows need to add to 1.

    Methods:
    --------
    fit():
        Fit an HMM model to provided data using Viterbi re-estimation (i.e. the EM algorithm).

    forward():
        Calculate the log-likelihood of the provided observation.

    load():
        Initialize an HMM model with a provided transition matrix and emission densities

    sample():
        Generate samples from the HMM

    viterbi():
        Calculate the optimal state sequence for the given observation
        sequence and given HMM model.

    Example (execute the class to run the example as a doctest)
    -----------------------------------------------------------
    >>> import numpy as np
    >>> from gaussian import Gaussian
    >>> signal1 = np.array([[ 1. ,  1.1,  0.9, 1.0, 0.0,  0.2,  0.1,  0.3,  3.4,  3.6,  3.5]])
    >>> signal2 = np.array([[0.8, 1.2, 0.4, 0.2, 0.15, 2.8, 3.6]])
    >>> data = np.hstack([signal1, signal2])
    >>> lengths = [11, 7]
    >>> hmm = HMM()
    >>> hmm.fit(data,lengths, 3)
    >>> trans, dists = hmm.trans, hmm.dists
    >>> means = [d.get_mean() for d in dists]
    >>> covs = [d.get_cov() for d in dists]
    >>> covs = np.array(covs).flatten()
    >>> means = np.array(means).flatten()
    >>> print(trans)
    [[ 0.66666667  0.33333333  0.          0.        ]
     [ 0.          0.71428571  0.28571429  0.        ]
     [ 0.          0.          0.6         0.4       ]
     [ 1.          0.          0.          0.        ]]
    >>> print(covs)
    [ 0.02        0.01702381  0.112     ]
    >>> print(means)
    [ 1.          0.19285714  3.38      ]
    >>> signal = np.array([[ 0.9515792,   0.9832767,   1.04633007,  1.01464327,  0.98207072,  1.01116689, 0.31622856,  0.20819263,  3.57707616]])
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)
    [0 0 0 0 0 0 1 1 2]
    >>> print(ll)
    2.23881485357
    >>> hmm.load(trans, dists)
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)
    [0 0 0 0 0 0 1 1 2]
    >>> print(ll)
    2.23881485357
    >>> print(hmm.score(signal, vals))
    2.23881485357
    >>> print(hmm.forward(signal))
    2.23882615241
    >>> signal = np.array([[ 0.9515792,   0.832767,   3.57707616]])
    >>> vals, ll = hmm.viterbi(signal)
    >>> print(vals)
    [0 1 2]
    >>> print(ll)
    -13.1960946635
    >>> samples, states = hmm.sample()
    '''

    def __init__(self, diagcov=True, maxiters=20, rtol=1e-4):
        '''
        Create an instance of the HMM class, with n_states hidden emitting states.

        Parameters
        ----------
        diagcov: boolean
            Indicates whether the Gaussians emission densities returned by training
            should have diagonal covariance matrices or not.
            diagcov = True, estimates diagonal covariance matrix
            diagcov = False, estimates full covariance matrix

        maxiters: int
            Maximum number of iterations used in Viterbi re-estimation
            Default: maxiters=20

        rtol: float
            Error tolerance for Viterbi re-estimation
            Default: rtol = 1e-4
        '''

        self.diagcov = diagcov
        self.maxiters = maxiters
        self.rtol = rtol

    def fit(self, data, lengths, n_states):
        '''
        Fit a left-to-right HMM model to the training data provided in `data`.
        The training data consists of l different observaion sequences,
        each sequence of length n_obs_i specified in `lengths`.
        The fitting uses Viterbi re-estimation (i.e. the EM algorithm).

        Parameters
        ----------
        data : (d,n_obs) ndarray
            An arrray of the trainining data, consisting of several different
            sequences.
            Note: Each observation has d features, and there are a total of n_obs
            observation.

        lengths: (l,) int ndarray
            Specifies the length of each separate observation sequence in `data`
            There are l difference training sequences.

        n_states : int
            The number of hidden emitting states to use initially.
        '''

        # Split the data into separate signals and pass to class
        self.data = data
        newstarts = np.cumsum(lengths)[:-1]
        self.signals = np.hsplit(data, newstarts)
        self.trans = HMM._ltrtrans(n_states)
        self.trans, self.dists, newLL, iters = self._em(self.trans, self._ltrinit())

    def load(self, trans, dists):
        '''
        Initialize an HMM model using the provided data.

        Parameters
        ----------
        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each
            emitting state.

        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1.

        '''

        self.trans, self.dists = trans, dists
        n_states = self.trans.shape[0] -1

    def _n_states(self):
        '''
        Get the number of emitting states used by the model.

        Return
        ------
        n_states : int
        The number of hidden emitting states to use initially.
        '''

        return self.trans.shape[0]-1

    def _n_obs(self):
        '''
        Get the total number of observations in all signals in the data associated with the model.

        Return
        ------
        n_obs: int
            The total number of observations in all the sequences combined.
        '''

        return self.data.shape[1]

    @staticmethod
    def _ltrtrans(n_states):
        '''
        Intialize the transition matrix (self.trans) with n_states emitting states (and an initial and
        final non-emitting state) enforcing a left-to-right topology.  This means
        broadly: no transitions from higher-numbered to lower-numbered states are
        permitted, while all other transitions are permitted.
        All legal transitions from a given state should be equally likely.

        The following exceptions apply:
        -The initial state may not transition to the final state
        -The final state may not transition (all transition probabilities from
         this state should be 0)

        Parameter
        ---------
        n_states : int
            Number of emitting states for the transition matrix

        Return
        ------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table initialized as described below.
        '''

        trans = np.zeros((n_states + 1, n_states + 1))
        trans[-1, :] = 1. / n_states
        for row in range(n_states):
            prob = 1./(n_states + 1 - row)
            for col in range(row, n_states+1):
                trans[row, col] = prob
        return trans

    def _ltrinit(self):
        '''
        Initial allocation of the observations to states in a left-to-right manner.
        It uses the observation data that is already available to the class.

        Note: Each signal consists of a number of observations. Each observation is
        allocated to one of the n_states emitting states in a left-to-right manner
        by splitting the observations of each signal into approximately equally-sized
        chunks of increasing state number, with the number of chunks determined by the
        number of emitting states.
        If 'n' is the number of observations in signal, the allocation for signal is specified by:
        np.floor(np.linspace(0, n_states, n, endpoint=False))

        Returns
        ------
        states : (n_obs, n_states) ndarray
            Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
            'states[:,j]' specifies the allocation of all the observations to state j.
        '''

        states = np.zeros((self._n_obs(), self._n_states()))
        i = 0
        for s in self.signals:
            vals = np.floor(np.linspace(0, self._n_states(), num=s.shape[1], endpoint=False))
            for v in vals:
                states[i][int(v)] = 1
                i += 1
        return np.array(states,dtype = bool)

    def viterbi(self, signal):
        '''
        See documentation for _viterbi()
        '''
        return HMM._viterbi(signal, self.trans, self.dists)

    @staticmethod
    def _viterbi(signal, trans, dists):
        '''
        Apply the Viterbi algorithm to the observations provided in 'signal'.
        Note: `signal` is a SINGLE observation sequence.

        Returns the maximum likelihood hidden state sequence as well as the
        log-likelihood of that sequence.

        Note that this function may behave strangely if the provided sequence
        is impossible under the model - e.g. if the transition model requires
        more observations than provided in the signal.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations

        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1.

        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each
            emitting  state.

        Return
        ------
        seq : (n,) ndarray
            The optimal state sequence for the signal (excluding non-emitting states)

        ll : float
            The log-likelihood associated with the sequence
        '''

        n_states = trans.shape[0] - 1
        T = signal.shape[1]   # T - number of observations
        lltable = np.zeros((n_states+1, T+1)) # table containing LLs of ML state sequence
        seq = np.zeros(T, dtype="int")
        backtable = np.zeros((n_states+1, T+1), dtype="int") # Back pointers for ML state sequence (0 is start)

        # Prepare time -1 column - overwritten with time n stuff later
        lltable[-1, -1] = 0
        for state in range(n_states):
            lltable[state, -1] = float("-inf")
        emissionLLs = np.array([[s.logf(x.flatten()) for s in dists] for x in np.hsplit(signal, signal.shape[1])])

        for time in range(T):
            lltable[n_states, time] = float("-inf")
            for state in range(n_states):
                bestpred = np.argmax(np.log(trans[:, state])+lltable[:, time-1]) # Underflow issue with log 0 - np.seterr to avoid
                lltable[state, time] = np.log(trans[bestpred, state]) + lltable[bestpred, time-1]
                lltable[state, time] += emissionLLs[time, state]
                backtable[state, time] = bestpred
        for state in range(n_states+1): # Last time step (t = n+1) - transition to final state
            bestpred = np.argmax(np.log(trans[:, state])+lltable[:, T-1])
            lltable[state, T] = lltable[bestpred, T-1] + np.log(trans[bestpred, state]) # No emission prob
            backtable[state, T] = bestpred
        seq[T-1] = backtable[n_states, T]
        for i in range(T-2, -1, -1):
            seq[i] = backtable[seq[i+1], i+1]
        return seq, lltable[n_states, T]

    def score(self, signal, seq):
        '''
        See documentation for _score()
        '''
        return HMM._score(signal, seq, self.trans, self.dists)

    @staticmethod
    def _score(signal, seq, trans, dists):
        '''
        Calculate the likelihood of an observation sequence and hidden state correspondence.
        Note: signal is a SINGLE observation sequence, and seq is the corresponding series of
        emitting states being scored.

        Returns the log-likelihood of the observation-states correspondence.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations

        seq : (n,) ndarray
            The state sequence provided for the signal (excluding non-emitting states)

        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1.

        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each
            emitting  state.

        Return
        ------
        ll : float
            The log-likelihood associated with the observation and state sequence under the model.
        '''
        # set initial variables (time and state)
        current_state = -1
        t = 0

        # get log loglik for first step
        # (this is simply the transition probability)
        loglik = np.log(trans[current_state][seq[0]])
        current_state = seq[0]

        # loop through sequence and get the log sum of the probabilities for
        # all states in the sequence
        for next_state in seq[1:]:
            transition_prob = np.log(trans[current_state][next_state])
            state_prob = dists[current_state].loglik(signal[:, t])
            loglik += transition_prob + state_prob

            current_state = next_state
            t += 1

        # do the last step (from state N-1 to state N)
        transition_prob = np.log(trans[current_state][current_state+1])
        state_prob = dists[current_state].loglik(signal[:, t])
        loglik += transition_prob + state_prob

        return loglik

    def forward(self, signal):
        '''
        See documentation for _forward()
        '''
        return HMM._forward(signal, self.trans, self.dists)

    @staticmethod
    def _forward(signal, trans, dists):
        '''
        Apply the forward algorithm to the observations provided in 'signal' to
        calculate its likelihood.
        Note: `signal` is a SINGLE observation sequence.

        Returns the log-likelihood of the observation.

        Parameters
        ----------
        signal : (d,n) ndarray
            Signal for which the optimal state sequence is to be calculated.
            d is the dimension of each observation (number of features)
            n is the number of observations

        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1.

        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each
            emitting  state.

        Return
        ------
        ll : float
            The log-likelihood associated with the observation under the model.
        '''
        # A non-log version
        # N = len(dists)
        # T = signal.shape[1]
        #
        # # setup step
        # current_a = np.array(trans[-1, :N])
        # next_a = np.zeros(N)
        #
        # # first step
        # for j in np.arange(N):
        #     current_dist = dists[j].likelihood(signal[:, 0])
        #     current_a[j] *= current_dist
        #
        # # loop through signal
        # for t in np.arange(1, T):
        #     for j in np.arange(N):
        #         current_dist =  dists[j].likelihood(signal[:, t])
        #         for i in np.arange(N):
        #             transition_prob = trans[i][j]
        #             next_a[j] += transition_prob * current_a[i]
        #         next_a[j] *= current_dist
        #     current_a = next_a
        #     next_a = np.zeros(N)
        #
        # ans = 0
        # for i in np.arange(N):
        #     transition_prob = trans[i][N]
        #     ans += transition_prob * current_a[i]
        # return np.log(ans)

        N = len(dists)
        T = signal.shape[1]

        # setup step
        current_a = np.log(np.array(trans[-1, :N]))
        next_a = np.zeros(N)

        # first step
        for j in np.arange(N):
            current_dist =  dists[j].loglik(signal[:, 0])
            current_a[j] += current_dist

        # loop through signal
        for t in np.arange(1, T):
            for j in np.arange(N):
                next_a[j] = -float('inf')
                current_dist = dists[j].loglik(signal[:, t])
                for i in np.arange(N):
                    transition_prob = np.log(trans[i][j])
                    next_a[j] = np.logaddexp(next_a[j], transition_prob + current_a[i])
                next_a[j] += current_dist
            current_a = next_a
            next_a = np.zeros(N)

        loglik = -float('inf')
        for i in np.arange(N):
            transition_prob = np.log(trans[i][N])
            loglik = np.logaddexp(loglik, transition_prob + current_a[i])
        return loglik

    def _calcstates(self, trans, dists):
        '''
        Calculate state sequences on the 'signals' maximizing the likelihood for
        the given HMM parameters.

        Calculate the state sequences for each of the given 'signals', maximizing the
        likelihood of the given parameters of a HMM model. This allocates each of the
        observations, in all the equences, to one of the states.

        Use the state allocation to calculate an updated transition matrix.

        IMPORTANT: As part of this updated transition matrix calculation, emitting states which
        are not used in the new state allocation are removed.

        In what follows, n_states is the number of emitting states described in trans,
        while n_states' is the new number of emitting states.

        Note: signals consists of ALL the training sequences and is available
        through the class.

        Parameters
        ----------
        trans : (n_states+1,n_states+1) ndarray
            The transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1.

        dists: (n_states,) list
            A list of Gaussian objects defining the emitting pdf's, one object for each
            emitting  state.

        Return
        ------
        states : bool (n_obs,n_states') ndarray
            The updated state allocations of each observation in all signals
        trans : (n_states'+ 1,n_states'+1) ndarray
            Updated transition matrix
        ll : float
            Log-likelihood of all the data
        '''

        maxstates = trans.shape[0]-1 # Exclude initial and final states
        newtrans = np.zeros_like(trans)
        used = [False] * maxstates + [True]
        states = np.zeros((self._n_obs(), maxstates))
        i, totalll = 0, 0
        for s in self.signals:
            seq, ll = HMM._viterbi(s, trans, dists)
            totalll += ll
            oldv = -1
            for v in seq:
                states[i][v] = 1
                newtrans[oldv, v] += 1
                oldv = v
                used[v] = True
                i += 1
            newtrans[seq[-1], -1] += 1
        newstates = np.where(used)[0]
        newtrans = newtrans[newstates, :]
        newtrans = newtrans[:, newstates]
        rowsums = np.sum(newtrans, axis=1) # Normalize counts for transition probabilities
        newtrans = newtrans/rowsums[:, np.newaxis]
        states = np.array(states[:, newstates[:-1]],dtype=bool) # Remove unused states by indexing
        return states, newtrans, totalll

    def _updatecovs(self, states):
        '''
        Update estimates of the means and covariance matrices for each HMM state

        Estimate the covariance matrices for each of the n_states emitting HMM states for
        the given allocation of the observations in self.data to states.
        If self.diagcov is true, diagonal covariance matrices are returned.

        Parameters
        ----------
        states : bool (n_obs,n_states) ndarray
            Current state allocations for self.data in model

        Return
        ------
        covs: (n_states, d, d) ndarray
            The updated covariance matrices for each state

        means: (n_states, d) ndarray
            The updated means for each state
        '''

        n_states = states.shape[1]
        d = self.data.shape[0]
        covs = np.zeros((n_states, d, d))
        means = np.zeros((n_states, d))
        for i in range(n_states):
            dati = self.data[:,states[:,i]]
            if dati.shape[1] > 0:
                means[i] = np.mean(dati,axis=1)
            else:
                warn("Attempting to calculate mean from no observations, setting to zero")
                means[i] = np.zeros(d)
            if dati.shape[1] > 1:
                covs[i] = np.array([np.cov(dati)],ndmin=2)
                if np.count_nonzero(covs[i]) == 0: # Degenerate covariance
                    # Dirty hack - better would be a Bayesian or regularization
                    # approach to smoothing the covariance matrix
                    warn("Zero covariance matrix obtained, setting to identity")
                    covs[i] = np.eye(dati.shape[0])
            else:
                warn("Attempting to calculate covariance matrix from one observation, setting to identity")
                covs[i] = np.eye(dati.shape[0])
        if self.diagcov:
            covs = np.array([np.diag(np.diag(c)) for c in covs])
        return covs, means

    def _em(self, trans, states):
        '''
        Perform parameter estimation for a hidden Markov model (HMM).

        Perform parameter estimation for an HMM using multi-dimensional Gaussian
        states.  The training observation sequences, signals,  are available
        to the class, and states designates the initial allocation of emitting states to the
        signal time steps.   The HMM parameters are estimated using Viterbi
        re-estimation.

        Note: It is possible that some states are never allocated any
        observations.  Those states are then removed from the states table, effectively redusing
        the number of emitting states. In what follows, n_states is the original
        number of emitting states, while n_states' is the final number of
        emitting states, after those states to which no observations were assigned,
        have been removed.

        Parameters
        ----------
        trans : (n_states+1,n_states+1) ndarray
            The left-to-right transition probability table.  The rightmost column contains probability
            of transitioning to final state, and the last row the initial state's
            transition probabilities.   Note that all the rows need to add to 1.

        states : (n_obs, n_states) ndarray
            Initial allocation of signal time-steps to states as a one-hot encoding.  Thus
            'states[:,j]' specifies the allocation of all the observations to state j.

        Return
        ------
        trans : (n_states'+1,n_states'+1) ndarray
            Updated transition probability table

        dists : (n_states',) list
            Gaussian object of each component.

        newLL : float
            Log-likelihood of parameters at convergence.

        iters: int
            The number of iterations needed for convergence
        '''

        covs, means = self._updatecovs(states) # Initialize the covariances and means using the initial state allocation
        dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
        oldstates, trans, oldLL = self._calcstates(trans, dists)
        converged = False
        iters = 0
        while not converged and iters <  self.maxiters:
            covs, means = self._updatecovs(oldstates)
            dists = [Gaussian(mean=means[i], cov=covs[i]) for i in range(len(covs))]
            newstates, trans, newLL = self._calcstates(trans, dists)
            if abs(newLL - oldLL) / abs(oldLL) < self.rtol:
                converged = True
            oldstates, oldLL = newstates, newLL
            iters += 1
        if iters >= self.maxiters:
            warn("Maximum number of iterations reached - HMM parameters may not have converged")
        return trans, dists, newLL, iters

    def sample(self):
        '''
        Draw samples from the HMM using the present model parameters. The sequence
        terminates when the final non-emitting state is entered. For the
        left-to-right topology used, this should happen after a finite number of
        samples is generated, modeling a finite observation sequence.

        Returns
        -------
        samples: (n,) ndarray
            The samples generated by the model
        states: (n,) ndarray
            The state allocation of each sample. Only the emitting states are
            recorded. The states are numbered from 0 to n_states-1.

        Sample usage
        ------------
        Example below commented out, since results are random and thus not suitable for doctesting.
        However, the example is based on the model fit in the doctests for the class.
        #>>> samples, states = hmm.samples()
        #>>> print(samples)
        #[ 0.9515792   0.9832767   1.04633007  1.01464327  0.98207072  1.01116689
        #  0.31622856  0.20819263  3.57707616]
        #>>> print(states)   #These will differ for each call
        #[1 1 1 1 1 1 2 2 3]
        '''

        #######################################################################
        import scipy.interpolate as interpolate
        def draw_discrete_sample(discr_prob):
            '''
            Draw a single discrete sample from a probability distribution.

            Parameters
            ----------
            discr_prob: (n,) ndarray
                The probability distribution.
                Note: sum(discr_prob) = 1

            Returns
            -------
            sample: int
                The discrete sample.
                Note: sample takes on the values in the set {0,1,n-1}, where
                n is the the number of discrete probabilities.
            '''

            if not np.sum(discr_prob) == 1:
                raise ValueError('The sum of the discrete probabilities should add to 1')
            x = np.cumsum(discr_prob)
            x = np.hstack((0.,x))
            y = np.array(range(len(x)))
            fn = interpolate.interp1d(x,y)
            r = np.random.rand(1)
            return np.array(np.floor(fn(r)),dtype=int)[0]
        #######################################################################

        #TODO: Using the function defined above, draw samples from the HMM
        from scipy.stats import multivariate_normal
        samples = np.array([0,0])
        states = np.array([])
        state = -1
        fin_state = self._n_states()

        state = draw_discrete_sample(self.trans[state])
        while state != fin_state:
            dist = self.dists[state]
            samples = np.vstack(
                [samples,
                multivariate_normal.rvs(
                    mean=dist.get_mean(),
                    cov=dist.get_cov()
                    )]
                )
            states = np.append(states, state)
            state = draw_discrete_sample(self.trans[state])

        samples = samples[1:]
        return [samples, states]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
