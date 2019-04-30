import qutip as qt
import numpy as np
import itertools

def quantize_potential_dft(U, xmin = None, xmax = None):
    if xmin == None:
        xmin = -np.pi*np.ones(U.ndim)
    if xmax == None:
        xmax = np.pi*np.ones(U.ndim)
    UF = np.fft.fftn(U)
    N = U.shape
    s = [2*np.pi*(1-1/Nn)/(xmaxn-xminn) for Nn,xmaxn,xminn in zip(N,xmax,xmin)]
    theta = [-sn*xminn for sn,xminn in zip(s,xmin)]
    Nq = [int((Nn-1)/2) if Nn%2 else int(Nn/2) for Nn in N]
    Ep = []
    for D in range(len(Nq)):
        Ep.append(qt.tensor([np.exp(1j*thetan)*qt.qdiags(np.ones(2*Nqn),-1)
                        if D==d
                        else qt.qeye(2*Nqn+1)
                        for d, (thetan,Nqn) in enumerate(zip(theta,Nq))]))
    I = qt.tensor([qt.qeye(2*Nqn +1) for Nqn in Nq])
    ops = [I for i in range(len(Nq))]
    k_last = np.zeros(np.shape(Nq), dtype=int)
    out = 0
    for k in np.ndindex(*[Nqn+1 for Nqn in Nq]):
        k_diff = np.array(list(k))-k_last
        for d in range(len(Nq)):
            if not k_diff[d] == 0:
                if k[d] == 0:
                    ops[d] = I
                else:
                    ops[d] = Ep[d]*ops[d]
                    if k[d] == N[d]/2:
                        ops[d] = ops[d] / 2 # Because otherwise nyquist freq is counted double
        pm_iter = [[1] if ki == 0 else [1,-1] for ki in k]
        for dir in itertools.product(*pm_iter):
            ck = UF[tuple(np.array(list(dir))*k)]/np.prod(N)
            out += ck*np.prod([ops[d] if sgn == 1 else ops[d].dag() for d,sgn in enumerate(dir)])
        k_last = np.array(list(k))
    return out, Nq
