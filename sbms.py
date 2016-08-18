from __future__ import division  # division returns always float
import numpy as np
from scipy.special import digamma 
from scipy.special import gammaln 
import fabsbm

def entropy(X):
    x = X[X > 0]
    return - np.sum(x * np.log(x))

class ICL_SBM(fabsbm.EM_SBM):
    """ICL [Daudin et al. Statistics and Computing, 2008].
    """
    def compute_lowerbound(self):
        return self.TLL() * self.N2 \
               - (1/4) * self.K * (self.K + 1) * np.log(self.N2) \
               - (1/2) * (self.K - 1) * np.log(self.N) \

class ICLO_SBM(ICL_SBM):
    """Corrected ICL (see our arxiv paper).
    """

    def compute_lowerbound(self):
        return ICL_SBM.compute_lowerbound(self) - self.N * entropy(self.gamma)

class FABVB_SBM(fabsbm.EM_SBM):
    """FAB algorithm of SBM.
    """
    def predict_Xij(self, i, j):
        return np.dot(np.dot(self.EZ[i, ], self.Pi), self.EZ[j, ])

    def do_Estep(self, itr=None, opt=None):
        Theta = fabsbm.logit(self.Pi)
        Psi = fabsbm.log1exp(Theta)
        D = self.K * (self.K + 1) / 2
        
        old_EZi = np.zeros(self.K)
        error = np.float('Inf')
        while error / self.N > 1e-6:#opt['threshold']:
            error = 0
            for i in np.random.permutation(self.N):
                old_EZi[:] = self.EZ[i, ]
                self.EZ[i, ] = np.log(self.gamma) \
                               + np.dot(Theta, np.dot(self.EZ.T, self.X[i, ])) \
                               - np.sum(np.dot(Psi, self.EZ.T), 1) \
                               - D / (2 * self.N * self.gamma)
                
                fabsbm.normalize_logprob(self.EZ[i, ])
                error += np.sum(np.abs(old_EZi - self.EZ[i, ]))

            self.update_gamma()
            del_ind = np.where(self.gamma <= 1e-20)[0]
            if len(del_ind) > 0:
                self.prune_group(del_ind)
                break
        self.update_SZZ()

    def update_SZZ(self):
        self.SZZ = np.dot(np.dot(self.EZ.T, self.X), self.EZ)
        
    def _update_Pi(self):
        self.Pi = self.SZZ / np.outer(self.N * self.gamma, self.N * self.gamma)


class VB_SBM(fabsbm.EM_SBM):
    """Variational EM algorithm of SBM [Latouche et al. Statistical Modelling, 2012]
    """
    def init_vars(self, X, K, init):
        fabsbm.EM_SBM.init_vars(self, X, K, init)
        
        self.n0    = 0.5 * np.ones(self.K)#init['n']
        self.Eta0  = 0.5 * np.ones([self.K] * 2)#init['Eta']
        self.Zeta0 = 0.5 * np.ones([self.K] * 2)#init['Zeta']

#        for i in xrange(self.N):
#            self.EZ[i, :] += np.random.rand(self.K)
#            self.EZ[i, ] /= np.sum(self.EZ[i, ])
    
    def do_Estep(self, itr=None, opt=None):
        #return
        Eta  = self.get_Eta()
        Zeta = self.get_Zeta()
        n = self.N * self.gamma
        de = digamma(Eta)
        dz = digamma(Zeta)
        dn = digamma(n) - digamma(np.sum(n))
        dzndez = dz - digamma(Eta + Zeta)
        dendz = de - dz

        old_EZi = np.zeros(self.K)
        error = np.float('Inf')
        while error / self.N > 1e-6:#opt['threshold']:
            error = 0
            for i in np.random.permutation(self.N):
                old_EZi[:] = self.EZ[i, ]
                self.EZ[i, ] = dn + np.dot(n - self.EZ[i, ], dzndez) \
                               + np.dot(np.dot(self.X[i, ], self.EZ), dendz)
                               #+ np.dot(np.sum(self.EZ[self.nb[i], ], 0), dendz)
                fabsbm.normalize_logprob(self.EZ[i, ])
                error += np.sum(np.abs(old_EZi - self.EZ[i, ]))

            del_ind = np.where(np.mean(self.EZ, 0) <= 1e-20)[0]
            if len(del_ind) > 0:
                self.prune_group(del_ind)
                break

    def predict_Xij(self, i, j):
        return np.dot(np.dot(self.EZ[i, ], self.Pi), self.EZ[j, ])

#    def update_SZZ(self):
#        self.SZZ[:] = 0
#        for m in xrange(self.NNZ):
#            (i, j) = self.m2ij(m)
#            self.SZZ += np.outer(self.EZ[i, ], self.EZ[j, ])

    def prune_group(self, del_ind):
        fabsbm.EM_SBM.prune_group(self, del_ind)
        self.Zeta0 = np.delete(self.Zeta0, del_ind, 0)
        self.Zeta0 = np.delete(self.Zeta0, del_ind, 1)
        self.Eta0 = np.delete(self.Eta0, del_ind, 0)
        self.Eta0 = np.delete(self.Eta0, del_ind, 1)
        self.n0 = np.delete(self.n0, del_ind, 0)

    def _update_Pi(self):
        self.Pi = self.get_Eta() / self.get_Zeta()

    def udpate_gamma(self):
        fabsbm.EM_SBM.update_gamma(self)
        self.gamma += self.n0 / self.N
        self.gamma /= np.sum(self.gamma)

    def get_Eta(self):
        #Eta = self.SZZ
        Eta = np.dot(np.dot(self.EZ.T, self.X), self.EZ)
        Eta[np.diag_indices(self.K)] /= 2
        return self.Eta0 + Eta

    def get_Zeta(self):
        Zeta = np.dot(np.dot(self.EZ.T, 1 - self.X - np.eye(self.N)), self.EZ)
        Zeta[np.diag_indices(self.K)] /= 2
        return self.Zeta0 + Zeta

    def compute_lowerbound(self):
        Eta  = self.get_Eta()
        Zeta = self.get_Zeta()
        n = self.N * self.gamma
        Eta0 = self.Eta0
        Zeta0 = self.Zeta0
        n0 = self.n0
        
        tri = np.triu_indices(self.K)
        return + (gammaln(np.sum(n0)) + np.sum(gammaln(n))) \
               - (gammaln(np.sum(n)) + np.sum(gammaln(n0))) \
               + np.sum((gammaln(Eta0 + Zeta0) + gammaln(Eta) + gammaln(Zeta))[tri]) \
               - np.sum((gammaln(Eta + Zeta) + gammaln(Eta0) + gammaln(Zeta0))[tri]) \
               + entropy(self.EZ)
    

if __name__ == "__main__":
    np.random.seed(2)
    _N = 200
    _K = 4
    _K_init = 6
    _X, _Pi, _a = fabsbm.generate_X(_N, _K, splitting='balanced')
    fabsbm.make_missing(_X, missing_ratio=0.1)
    verbose = 1
    
    #sbm = ICL_SBM(verbose)
    #sbm = ICLO_SBM(verbose)    
    #sbm = VB_SBM(verbose)
    sbm = FABVB_SBM(verbose)    
    sbm.train(_X, _K_init, max_itr=64)

#    models = [None]
#    for k in xrange(1, _K_init + 1):
#        models.append(VB_SBM(verbose))
#        models[k].train(_X, k, max_itr=32)
#    
#    print [models[k].compute_lowerbound() for k in xrange(1, _K_init + 1)]
