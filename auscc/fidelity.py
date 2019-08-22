from qutip import ket2dm, expect, parallel_map, Qobj, qeye
from numpy import roll, exp, pi, sqrt, zeros, real, imag, abs, sum
from numpy.linalg import inv
import warnings

def build_unitary_basis(subspace_basis):
    d = len(subspace_basis)
    X = sum([psi1 * psi2.dag() for psi1,psi2 in zip(subspace_basis, roll(subspace_basis, 1))])
    Z = sum([exp(2*pi*1j*n/d)*psi*psi.dag() for n,psi in enumerate(subspace_basis)])
    u_basis = []
    for l in range(d):
        for k in range(d):
            u_basis.append(X**k * Z**l)
    return u_basis

def build_pure_state_basis(subspace_basis, u_basis):
    d = len(subspace_basis)
    rho_basis = []
    ket_basis = []
    for k in range(d):
        for l in range(k,d):
            if k == l:
                ket_basis.append(subspace_basis[k])
            else:
                ket_basis.append((subspace_basis[k] + subspace_basis[l])/sqrt(2))
                ket_basis.append((subspace_basis[k] + 1j*subspace_basis[l])/sqrt(2))
    Tinv = zeros((d**2,d**2), dtype=complex)
    for k,ket in enumerate(ket_basis):
        for j,u in enumerate(u_basis):
            Tinv[j][k] = (u.dag() * ket2dm(ket)).tr()/d
    return ket_basis, inv(Tinv)

def leakage(channel, subspace_basis, tlist):
    d = len(subspace_basis)
    I_subspace = sum([ket2dm(ket) for ket in subspace_basis])
    return [1 - real(expect(I_subspace, rho)) for rho in channel(I_subspace/d, tlist)]

def entanglement_fidelity_term(k, channel, pure_state_basis, tlist, u_basis, d, T):
    e_ops_k = sum([T[k][j]*u.dag() / d**3 for j,u in enumerate(u_basis)])
    return [expect(e_ops_k, rho) for rho in channel(pure_state_basis[k], tlist)]

def entanglement_fidelity(channel, subspace_basis, tlist, progress_bar=False):
    d = len(subspace_basis)
    u_basis = build_unitary_basis(subspace_basis)
    pure_state_basis, T = build_pure_state_basis(subspace_basis, u_basis)
    if progress_bar:
        F_e = sum(parallel_map(entanglement_fidelity_term, range(d**2), (channel, pure_state_basis, tlist, u_basis, d, T), progress_bar=True), axis=0)
    else:
        F_e =sum(parallel_map(entanglement_fidelity_term, range(d**2), (channel, pure_state_basis, tlist, u_basis, d, T)), axis=0)
    if any(imag(F_e)>10**-12):
        warnings.warn('I got a pretty large imaginary number when calculating entanglement fidelity. Are you sure your simulation function is correct?')
    return real(F_e)

def average_fidelity(channel, subspace_basis,  tlist = None, progress_bar=False):
    """Calculates the average fidelity of a quantum channel.

    Parameters
    ----------
    channel : function
        The quantum channel we wish to find the average fidelity of. It should take (rho,tlist) as inputs and return eps(rho) to each time in tlist, where eps is the channel. If we want to calculate the average fidelity of some gate, U, the assosciated channel is U.dag()*(eps(rho))*U, that is, we let the system evolve (eps), and then apply the inverse gate operation
    subspace_basis : list
        List of basis kets, which describes logical subspace. If the channel sends a state out of this subspace it is counted as leakage.
    tlist : iterable
        Times at which the average fidelity is calculated.
    progress_bar : bool
        Determines if progressbar is shown.

    Returns
    -------
    numpy array
        Average fidelity of channel at times in tlist.

    """

    d = len(subspace_basis)
    L = leakage(channel, subspace_basis, tlist)
    F_e = entanglement_fidelity(channel, subspace_basis, tlist, progress_bar)
    return [(d*f_e + 1 - l) / (d + 1) for f_e,l in zip(F_e, L)]

def average_fidelity_pedersen(Kraus_ops, U_target = None, P = None):
    # Ensuring every input is put on correct form
    if isinstance(Kraus_ops, Qobj):
        Kraus_ops = [[Kraus_ops]]
    elif isinstance(Kraus_ops[0], Qobj):
        Kraus_ops = [Kraus_ops]
    if U_target == None:
        U_target = len(Kraus_ops)*[qeye(Kraus_ops[0][0].dims[0])]
    if isinstance(U_target,Qobj):
        U_target = len(Kraus_ops)*[U_target]
    if P == None:
        P = qeye(Kraus_ops[0][0].dims[0])
    n_rel = P.tr()
    F = zeros(len(Kraus_ops))
    # Calculating average fidelity
    for ind, G_list, U0 in zip(range(len(Kraus_ops)), Kraus_ops, U_target):
        for G in G_list:
            M = P*U0.dag()*G*P
            F[ind] += ( ( M.dag() * M ).tr()+abs( M.tr() )**2 ) / ( n_rel * ( n_rel + 1 ) )
    return F
