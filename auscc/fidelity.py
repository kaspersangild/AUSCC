import qutip as qt
import numpy as np

def build_unitary_basis(subspace_basis):
    d = len(subspace_basis)
    X = sum([psi1 * psi2.dag() for psi1,psi2 in zip(subspace_basis, np.roll(subspace_basis, 1))])
    Z = sum([np.exp(2*np.pi*1j*n/d)*psi*psi.dag() for n,psi in enumerate(subspace_basis)])
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
                ket_basis.append((subspace_basis[k] + subspace_basis[l])/np.sqrt(2))
                ket_basis.append((subspace_basis[k] + 1j*subspace_basis[l])/np.sqrt(2))
    Tinv = np.zeros((d**2,d**2), dtype=complex)
    for k,ket in enumerate(ket_basis):
        for j,u in enumerate(u_basis):
            Tinv[j][k] = (u.dag() * qt.ket2dm(ket)).tr()/d
    return ket_basis, np.linalg.inv(Tinv)

def leakage(process, subspace_basis, tlist):
    d = len(subspace_basis)
    I_subspace = sum([qt.ket2dm(ket) for ket in subspace_basis])
    return [1 - e for e in process(I_subspace/d, tlist, I_subspace)]

def entanglement_fidelity_term(k, process, pure_state_basis, tlist, u_basis, d, T):
    e_ops_k = sum([T[k][j]*u.dag() / d**3 for j,u in enumerate(u_basis)])
    return process(pure_state_basis[k], tlist, e_ops_k)

def entanglement_fidelity(process, subspace_basis, tlist, progress_bar=False):
    d = len(subspace_basis)
    u_basis = build_unitary_basis(subspace_basis)
    pure_state_basis, T = build_pure_state_basis(subspace_basis, u_basis)
    if progress_bar:
        F_e =np.sum(qt.parallel_map(entanglement_fidelity_term, range(d**2), (process, pure_state_basis, tlist, u_basis, d, T), progress_bar=True), axis=0)
    else:
        F_e =np.sum(qt.parallel_map(entanglement_fidelity_term, range(d**2), (process, pure_state_basis, tlist, u_basis, d, T)), axis=0)
    return F_e

def average_fidelity(process, subspace_basis,  tlist, progress_bar=False):
    d = len(subspace_basis)
    L = leakage(process, subspace_basis, tlist)
    F_e = entanglement_fidelity(process, subspace_basis, tlist)
    return [(d*f_e + 1 - l) / (d + 1) for f_e,l in zip(F_e, L)]
