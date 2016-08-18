import numpy as np
import sys
import datetime
import fabsbm
import sbms

datatype = sys.argv[1]
_N = int(sys.argv[2])
_K = int(sys.argv[3])
np.random.seed(int(sys.argv[4]))

_K_init = 20
_X, _Pi, _a = fabsbm.generate_X(_N, _K, splitting=datatype)
verbose = 0
max_itr = 256

FABs = set(('EM', 'VAB', 'FVAB', 'FABVB'))

if sys.argv[5] in FABs:
    if sys.argv[5] == 'EM':
        sbm = fabsbm.EM_SBM(verbose)
    elif sys.argv[5] == 'VAB':
        sbm = fabsbm.VAB_SBM(verbose)
    elif sys.argv[5] == 'FVAB':
        sbm = fabsbm.FVAB_SBM(verbose)
    elif sys.argv[5] == 'FABVB':
        sbm = sbms.FABVB_SBM(verbose)

    opt = dict(max_itr_BP=10, conv_thresh_BP=1e-2, start_penalty=1)
    sbm.train(_X, _K_init, max_itr=max_itr, Estep_opt=opt)
    runtime = sbm.runtime
    bestK = sbm.K

else:
    if sys.argv[5] == 'ICL':
        sbmc = sbms.ICL_SBM
    elif sys.argv[5] == 'ICLO':
        sbmc = sbms.ICLO_SBM
    elif sys.argv[5] == 'VB':
        sbmc = sbms.VB_SBM
        
    lb = np.ones(_K_init + 1) * np.float('-Inf')
    models = [None] * (_K_init + 1)
    runtime = 0
    for k in xrange(1, _K_init + 1):
        models[k] = sbmc(verbose)
        models[k].train(_X, k, max_itr=max_itr)
        lb[k] = models[k].compute_lowerbound()
        runtime += models[k].runtime
    bestK = np.argmax(lb)
    sbm = models[bestK]

print ' '.join(sys.argv[1:]), bestK, runtime, sbm.TLL(), sbm.PLL()
