import numpy as np

class MDS(object):

    def __init__(self):
        self.D = None
        self.target_d = None

    def fit(self, D):
        '''
        :param D: Distance matrix
        :return: None
        '''
        # Distances are squared
        self.D = np.array(D)**2
        # Double centerization
        s = -0.5*(self.D-np.mean(self.D, 0)-np.mean(self.D, 1).reshape((-1,1))+np.mean(self.D))
        evecs, evals, temp = np.linalg.svd(s)
        idx = np.argsort(evals)[::-1]
        self.evals, self.evecs = evals[idx], evecs[:,idx]
        self.n = len(self.D)

    def transform(self, p):
        '''
        :param p: Target number of dimensionality
        :return: Configuration matrix. Each raw is a transformed observation.
        '''
        evals_sqrt = np.sqrt(self.evals)
        return self.evecs[:,:p].dot(np.diag(evals_sqrt[:p])), self.evals