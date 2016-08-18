from __future__ import division  # division returns always float
import numpy as np
import scipy.linalg
import scipy.cluster
import scipy.misc
import scipy as sp
import datetime


def clipping_0to1(x, minval):
    x[x < minval] = minval
    x[1 - x < minval] = minval

def to_seconds_float(timedelta):
    """Calculate floating point representation of combined
    seconds/microseconds attributes in :param:`timedelta`.

    :raise ValueError: If :param:`timedelta.days` is truthy.

        >>> to_seconds_float(datetime.timedelta(seconds=1, milliseconds=500))
        1.5
    """
    return timedelta.seconds + timedelta.microseconds / 1E6 \
        + timedelta.days * 86400

def logit(x):
    return sp.special.logit(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log1exp(x):
    return np.log1p(np.exp(x))

def H_bernoulli(eta):
    mean = sigmoid(eta)
    return -np.nansum(mean * eta - log1exp(eta))

def mlogit(x):
    return np.log(x / x[-1])

def msigmoid(x):
    return  np.exp(x) / np.sum(np.exp(x))

def normalize_logprob(arr):
    arr -= sp.misc.logsumexp(arr)
    arr[:] =  np.exp(arr)


def Msigmoid(X):
    Ans = np.zeros(X.shape)
    for i in xrange(X.shape[0]):
        Ans[i, :] = msigmoid(X[i, :])
    return Ans

def generate_X(N, K, pattern=['diag', 'random'][0], splitting=['balanced', 'unbalanced'][0]):
    """Generating an adjacency matrix that follows SBM.

    Args:
    	N (int): # of nodes
        K (int): # of clusters
        pattern (str): type of Pi. If pattern='diag', Pi is diagonal.
        splitting (str): type of Gamma. If splitting='balanced', gamma_k = 1/K for k in [K].
    """
    X = np.zeros((N, N))
    if pattern == 'random':
        Pi = np.random.rand(K, K)
        Pi = (Pi + Pi.T) / 2
    elif pattern == 'diag':
        Pi = np.ones((K, K)) * (1 / N) 
        Pi[np.diag_indices(K)] *= 20
        
    if splitting == 'balanced':
        a = np.ones(K) 
    elif splitting == 'unbalanced':
        a = np.arange(K, dtype=np.float) + 1
    a /= np.sum(a)

    Nk = [int(N * np.sum(a[:k])) for k in xrange(K + 1)]
    for k in xrange(K):
        for l in xrange(k, K):
            submat = np.random.binomial(1, Pi[k, l], (Nk[k+1] - Nk[k], Nk[l+1] - Nk[l]))
            if k == l:
                L = submat.shape[0]
                submat[np.triu_indices(L)] = submat.T[np.triu_indices(L)]
                submat[np.diag_indices(L)] = 0
                
            X[Nk[k]:Nk[k+1], Nk[l]:Nk[l+1]] = submat
            if k != l:
                X[Nk[l]:Nk[l+1], Nk[k]:Nk[k+1]] = submat.T

    for i in np.where(np.sum(X, 1) == 0)[0]:
        j = np.random.choice(N, size=1)
        if i == j:
            j += 1
        X[i, j] = X[j, i] = 1

    return X, Pi, a

def make_missing(X, missing_ratio=0):
    ind = np.where(X == 1)
    nind = np.where(X == 0)
    M = int(len(ind[0]) / 2 * missing_ratio)
    
    for m in np.random.choice(len(ind[0]), int(M / 2), replace=False):
        (i, j) = (ind[0][m], ind[1][m])
        X[i,j] = X[j,i] = np.float('Inf')

    for m in np.random.choice(len(nind[0]), int(M / 2), replace=False):
        (i, j) = (nind[0][m], nind[1][m])
        X[i,j] = X[j,i] = np.float('-Inf')

def make_missing_unbiased(X, missing_ratio=0):
    N = X.shape[0]
    N2 = int(N * (N - 1) / 2)
    ind = np.triu_indices(N, 1)
    
    for m in np.random.choice(N2, int(N2 * missing_ratio), replace=False):
        (i, j) = (ind[0][m], ind[1][m])
        X[i,j] = X[j,i] = np.float('Inf') * (X[i, j] - 0.5)



class EM_SBM(object):
    """The EM algorithm of SBM. E-step is done by belief propagation (BP).
    """
    def __init__(self, verbose=1):
        self.verbose = verbose
        self.minval = 1e-40
        self.m_count = 0
        self.hard_assignments = []
        pass

    def train(self, X, init_K, init=None, max_itr=64, Pi_err_thresh=1e-8,
              Estep_opt=dict(max_itr_BP=10, conv_thresh_BP=1e-2, start_penalty=1),
              time_limit=60*60*24, log_cluster=False):
        self.start_time = datetime.datetime.now()
        self.stop_time = datetime.timedelta(0)
        self.init_vars(X, init_K, init)
        self.prune_group(del_ind=np.where(self.gamma <= 0)[0])
        self.do_Estep(-1, opt=dict(max_itr_BP=100, conv_thresh_BP=1e-2, start_penalty=1))

        self.print_error(-1)
        for itr in xrange(max_itr):
            self.impute_X()
            self.do_Estep(itr, Estep_opt)
            self.do_Mstep()

            if log_cluster:
                self.hard_assignments.append(np.argmax(self.EZ, axis=1))

            if np.log2(itr + 1) == int(np.log2(itr + 1)):
                self.print_error(itr)
                pass

            if self.Pi_error() < Pi_err_thresh:
                break

            if self.get_runtime_in_sec() > time_limit:
                self.runtime = np.nan
                return -1
        self.runtime = self.get_runtime_in_sec()

    def get_runtime_in_sec(self):
        return to_seconds_float(datetime.datetime.now() - self.start_time - self.stop_time)

    def impute_X(self):
        for m in xrange(self.NNA):
            (i, j) = self.m2ij(m, self.mind)
            self.X[i, j] = self.predict_Xij(i, j)

    def predict_Xij(self, i, j):
        return np.sum(self.EZZ[i, j] * self.Pi)

    def do_Estep(self, itr, opt):
        off_penalty = itr < opt['start_penalty']
        for itr_BP in xrange(opt['max_itr_BP']):
            self.update_EZ(off_penalty)
            self.update_gamma()
            conv, is_pruned = self.do_BP(off_penalty, prune_thresh=(1 / self.N) * 0.1)
            if is_pruned:
                continue
            if conv < opt['conv_thresh_BP']:
                break
        self.update_SZZ()

    def do_Mstep(self):
        self.update_gamma()
        self.update_Pi()

    def m2ij(self, m, ind=None):
        if ind is None:
            return (self.ind[0][m], self.ind[1][m])
        else:
            return (ind[0][m], ind[1][m])

    def init_vars(self, X, K, init):
        self.X = X
        self.N = X.shape[0]
        self.N2 = self.N * (self.N - 1)
        self.K = K
        self.orig_kind = np.arange(self.K)
        self.init_K = K
        
        self.ind = np.where(X == 1)
        self.NNZ = len(self.ind[0])
        self.nb  = [np.where(X[i, :] == 1)[0] for i in xrange(self.N)]
        self.mind = np.where(np.isinf(X))
        self.NNA = len(self.mind[0])
        self.mnb  = [np.where(np.isinf(X[i, :]))[0] for i in xrange(self.N)]
        self.indmind = np.nonzero(X)

        self.true_label = dict(zip([(self.mind[0][m], self.mind[1][m]) \
                                    for m in xrange(self.NNA)],  X[self.mind] > 0))

        self.gamma = np.ones(self.K) / self.K
        self.SZZ = np.zeros([self.K] * 2) * self.NNZ / (self.K ** 2)
        self.EZ  = np.zeros((self.N, self.K))
        self.h = self.N * np.ones(self.K) * (self.NNZ / self.N ** 2)
        
        self.M = {}
        self.EZZ = {}
        for m in xrange(self.NNZ + self.NNA):
            (i, j) = self.m2ij(m, self.indmind)
            self.M[(i, j)] = np.random.rand(self.K) + self.minval
            self.M[(i, j)] /= np.sum(self.M[(i, j)])
            self.EZZ[(i, j)] = np.outer(self.M[(i, j)], self.M[(i, j)])
            self.EZZ[(i, j)] /= np.sum(self.EZZ[(i, j)])
#            self.SZZ += self.EZZ[(i, j)]

        self.kmeans_Pi_and_M(X)
        self.Pi_old = np.zeros([self.K] * 2)


    def kmeans_Pi_and_M(self, X):
        X[np.isinf(X)] = self.NNZ / self.N2

        m = 1 / np.sqrt(np.sum(X, 1))
        Laplacian = (X * m).T
        Laplacian *= m
        Laplacian = np.eye(self.N) - Laplacian
        _, vec = sp.linalg.eigh(Laplacian, eigvals=(0, self.K))
        
        whitened = sp.cluster.vq.whiten(vec)
        centroid, _ = sp.cluster.vq.kmeans(whitened, self.K)
        label, _ = sp.cluster.vq.vq(whitened, centroid)

        self.Pi = np.zeros([self.K] * 2)
        for k in xrange(self.K):
            if np.sum(label == k) == 0:
                continue
            for l in xrange(self.K):
                if np.sum(label == l) == 0:
                    continue
                self.Pi[k, l] = np.mean(X[np.ix_(label == k, label == l)])
        clipping_0to1(self.Pi, self.N ** -2)

        self.EZ = np.array([label == k for k in xrange(self.K)], dtype=np.float).T
        for m in xrange(self.NNZ + self.NNA):
            (i, j) = self.m2ij(m, self.indmind)
            self.M[(i, j)] += self.EZ[i, ]
            self.M[(i, j)] /= np.sum(self.M[(i, j)])
        self.update_gamma()

    def update_EZ(self, off_penalty):
        for i in xrange(self.N):
            self.EZ[i, ] = self.log_unnormalized_message(i, off_penalty)
            normalize_logprob(self.EZ[i, ])

    def update_h(self):
        self.h = np.dot(self.Pi, np.sum(self.EZ, 0))

    def do_BP(self, off_penalty, prune_thresh):
        """ BP algorithm. Basically the same implementation of [Decelle et al. PRE, 2011]
        """
        self.update_h()
        conv = 0
        Mij_old = np.zeros(self.K)
        for m in np.random.permutation(self.NNZ + self.NNA):
            if self.verbose == 2:
                self.print_log(freq=10)
            
            (i, j) = self.m2ij(m, self.indmind)
            lmi = self.log_unnormalized_message(i, off_penalty)

            Mij_old[:] = self.M[(i, j)]
            self.M[(i, j)] = lmi - (self.X[i,j] * np.log(np.dot(self.M[(j, i)], self.Pi))) \
                                    + (1 - self.X[i,j]) * np.log(np.dot(self.M[(j, i)], (1 - self.Pi)))
            normalize_logprob(self.M[(i, j)])
            clipping_0to1(self.M[(i, j)], self.minval)
            conv += np.mean(np.abs(Mij_old - self.M[(i, j)]))

            self.h -= np.dot(self.Pi, self.EZ[i, ])
            self.gamma -= self.EZ[i, ] / self.N
            self.EZ[i, ] = self.log_unnormalized_message(i, off_penalty)
            normalize_logprob(self.EZ[i, ])
            self.h += np.dot(self.Pi, self.EZ[i, ])
            self.gamma += self.EZ[i, ] / self.N
            clipping_0to1(self.gamma, self.minval)

            self.EZZ[(i, j)] = (self.Pi ** self.X[i,j]) * ((1 - self.Pi) ** (1 - self.X[i,j])) \
                                    * np.outer(self.M[(i, j)], self.M[(j, i)])
            self.EZZ[(i, j)] /= np.sum(self.EZZ[(i, j)])

            del_ind = np.where(self.gamma <= prune_thresh)[0]
            if len(del_ind) > 0:
                self.prune_group(del_ind)
                break
        return conv, len(del_ind) > 0

    def log_unnormalized_message(self, i, off_penalty):
        #lmhup = np.sum(np.log(1 - np.dot(self.EZ[self.nnb[i], ], self.Pi)), 0)
        lmhup = -self.h
        lohup = 0
        
        for s in self.nb[i]:
            lohup += np.log(np.dot(self.M[(s, i)], self.Pi))
        for s in self.mnb[i]:
            lohup += np.log(np.dot(self.M[(s, i)], (self.Pi ** self.X[s, i]) \
                                   * ((1 - self.Pi) ** (1 - self.X[s, i]))))
        return np.log(self.gamma) + lmhup + lohup

    def prune_group(self, del_ind):
        if len(del_ind) == 0:
            return
        if self.K == 1:
            return
        self.EZ = np.delete(self.EZ, del_ind, 1)
        for m in xrange(self.NNZ + self.NNA):
            ij = self.m2ij(m, self.indmind)
            self.M[ij] = np.delete(self.M[ij], del_ind, 0)
            self.M[ij] /= np.sum(self.M[ij])
            self.EZZ[ij] = np.delete(self.EZZ[ij], del_ind, 0)
            self.EZZ[ij] = np.delete(self.EZZ[ij], del_ind, 1)
            self.EZZ[ij] /= np.sum(self.EZZ[ij])
        self.Pi = np.delete(self.Pi, del_ind, 0)
        self.Pi = np.delete(self.Pi, del_ind, 1)
        self.Pi_old = np.delete(self.Pi_old, del_ind, 1)
        self.Pi_old = np.delete(self.Pi_old, del_ind, 0)
        self.SZZ = np.delete(self.SZZ, del_ind, 0)
        self.SZZ = np.delete(self.SZZ, del_ind, 1)
        self.gamma = np.delete(self.gamma, del_ind, 0)
        self.h = np.delete(self.h, del_ind, 0)
        self.orig_kind = np.delete(self.orig_kind, del_ind, 0)
        self.K -= len(del_ind)

        if self.verbose == 1:
            print 'prune nodes',del_ind

    def update_gamma(self):
        self.gamma = np.mean(self.EZ, 0)
        clipping_0to1(self.gamma, self.minval)

    def update_SZZ(self):
        self.SZZ[:] = 0
        for m in xrange(self.NNZ):
            self.SZZ += self.EZZ[self.m2ij(m)]
        for m in xrange(self.NNA):
            (i, j) = self.m2ij(m, self.mind)
            self.SZZ += self.X[i, j] * self.EZZ[(i, j)]

    def update_Pi(self):
        self.Pi_old[:] = self.Pi
        self._update_Pi()
        clipping_0to1(self.Pi, self.N ** -2)

    def _update_Pi(self):
        self.Pi = self.SZZ / np.outer(self.N * self.gamma, self.N * self.gamma)

    def compute_lowerbound(self):
        single = 0
        for i in xrange(self.N):
            single += sp.misc.logsumexp(self.log_unnormalized_message(i, False))
            #single += sp.misc.logsumexp(self.log_unnormalized_message(i, True))
            
        joint = self.NNZ * np.log(self.N)
        for m in xrange(self.NNZ):
            (i, j) = self.m2ij(m)
            joint += np.log(np.dot(self.M[(i, j)], np.dot(self.Pi, self.M[(j, i)]))) 

        return (joint - single) / self.N \
               - self.N * np.dot(self.gamma, np.dot(self.Pi, self.gamma)) / 2

    def print_log(self, freq):
        self.m_count += 1
        wc = to_seconds_float(datetime.datetime.now() - self.start_time - self.stop_time)
        stop = datetime.datetime.now()
        if self.m_count % freq == 0:
            print wc, self.K, self.TLL(),
            for k in xrange(self.init_K):
                l = np.where(self.orig_kind == k)[0]
                if l.size == 0:
                    print 0,
                else:
                    print self.gamma[l[0]],
            print
        self.stop_time += datetime.datetime.now() - stop

    def TLL(self, X=None): # training log-likelihood
        ll = 0
        LP = np.log(self.Pi)
        LN = np.log(1 - self.Pi)
        if X is None:
            X = self.X
        for i in xrange(self.N - 1):
            for j in xrange(i + 1, self.N):
                if X[i, j] == 1:
                    ll += np.dot(np.dot(self.EZ[i, :], LP), self.EZ[j, :])
                elif X[i, j] == 0:
                    ll += np.dot(np.dot(self.EZ[i, :], LN), self.EZ[j, :])
                
        return ll / (self.N2 * 2)

    def PLL(self): # predictive (test) log-likelihood
        if self.NNA == 0:
            return 0
        ll = 0
        LP = np.log(self.Pi + self.minval)
        LN = np.log(1 - self.Pi + self.minval)
        for m in xrange(self.NNA):
            ij = self.m2ij(m, self.mind)
            if self.true_label[ij] == True:
                ll += np.sum(LP * self.EZZ[ij])
            else:
                ll += np.sum(LN * self.EZZ[ij])
                
        return ll / self.NNA


    def print_error(self, itr):
        if self.verbose == 1:
            print 'itr=',itr
            print self.compute_lowerbound(), self.TLL(), self.PLL(), self.Pi_error()
            print self.gamma
            print self.Pi

    def Pi_error(self):
        return np.max(np.abs(self.Pi - self.Pi_old))



class VAB_SBM(EM_SBM):
    """FIC+BP algorithm of SBM.
    """
    def log_unnormalized_message(self, i, off_penalty, coef=None):
        lm = EM_SBM.log_unnormalized_message(self, i, off_penalty)
        if not off_penalty:
            if coef is None:
                coef = self.K + 1
            Ez_quoti = self.N * self.gamma - self.EZ[i, ]
            #Ez_quoti += 1
            Ez_quoti[Ez_quoti < self.minval] = self.minval
            lm -= coef / 2 * np.log(1 + 1 / Ez_quoti)

        return lm


class FVAB_SBM(VAB_SBM):
    """F2AB algorithm of SBM.
    """

    def log_unnormalized_message(self, i, off_penalty):
        lm = VAB_SBM.log_unnormalized_message(self, i, off_penalty, 1)
        if not off_penalty:
            Zni = np.sum(self.EZ[self.nb[i], ], 0)
            O = np.outer(Zni, self.EZ[i, ])
            denom = self.NNZ * np.outer(self.gamma, self.gamma) - (O + O.T)
            #denom += np.ones((self.K, self.K))
            denom[denom < 1 / self.NNZ] = 1 / self.NNZ

            A = np.outer(Zni, np.ones(self.K))
            lm -= 0.5 * np.sum(np.log(1 + (A + A.T) / denom), 0)

        return lm



if __name__ == "__main__":
    np.random.seed(2)
    _N = 200
    _K = 4
    _K_init = 10
    _X, _Pi, _a = generate_X(_N, _K, splitting='balanced')
    make_missing(_X, missing_ratio=0.1)
    verbose = 1
    
    #sbm = EM_SBM(verbose)
    #sbm = VAB_SBM(verbose)
    #sbm = VAB2_SBM(verbose)
    sbm = FVAB_SBM(verbose)
    
    #sbm.train(_X, _K, init=dict(Pi=_Pi))
    sbm.train(_X, _K_init, max_itr=2, log_cluster=True)
    print sbm.runtime, sbm.PLL()
    #print np.vstack(sbm.hard_assignments)
