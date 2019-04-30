from auscc.fidelity import average_fidelity, entanglement_fidelity, leakage
from auscc.transform import transform_state_unitary
import qutip as qt
import numpy as np
# One qubit simple test unitary basis
basis = [qt.ket([0],[2]), qt.ket([1],[2])]
U_target = -1j*qt.sigmax()
H = qt.sigmax()
def process(rho, tlist):
    res = qt.mesolve(H, rho, tlist)
    return [transform_state_unitary(rho, U_target, inverse = True) for rho in res.states]
tlist = np.linspace(0,np.pi/2,10)
F_av = average_fidelity(process, basis, tlist)
print(F_av,'\n')
# One qubit with third state [0,1] comp basis + some noise
def projection_operator(subspace_basis, subspace_dims):
    d = len(subspace_basis)
    for j,ket in enumerate(subspace_basis):
        ket_subspace = qt.basis(d, j)
        if j == 0:
            P = ket_subspace*ket.dag()
        else:
            P+= ket_subspace*ket.dag()
    P.dims[0] = subspace_dims
    return P
subspace_basis =  [qt.ket([0], 3), qt.ket([1], 3) ]
P = projection_operator(subspace_basis, [2])
U_target = P.dag()*(-1j*qt.sigmax())*P
H = qt.create(3)+qt.destroy(3)+20*qt.ket([2],[3])*qt.bra([2],[3])
def process(rho, tlist):
    res = qt.mesolve(H, rho, tlist, c_ops = [0.1*qt.destroy(3)])
    return [transform_state_unitary(rho, U_target, inverse = True) for rho in res.states]
tlist = np.linspace(0,np.pi/2,10)
F_av = average_fidelity(process, subspace_basis, tlist)
print(F_av,'\n')
# One qudit with four states [1,2] comp basis
subspace_basis =  [qt.ket([1], 4), qt.ket([2], 4) ]
P = projection_operator(subspace_basis, [2])
U_target = P.dag()*(-1j*qt.sigmax())*P
H = qt.create(4)+qt.destroy(4)+20*qt.ket([3],[4])*qt.bra([3],[4])+20*qt.ket([0],[4])*qt.bra([0],[4])
def process(rho, tlist):
    res = qt.mesolve(H, rho, tlist)
    return [transform_state_unitary(rho, U_target, inverse = True) for rho in res.states]
tlist = np.linspace(0,np.pi/(2*np.sqrt(2)),10)
F_av = average_fidelity(process, subspace_basis, tlist)
print(F_av,'\n')
# One qubit doing nothing in interaction pic but simulated in schrodinger
subspace_basis = [qt.ket(seq,[2]) for seq in qt.state_number_enumerate([2])]
H0 = qt.sigmaz()
def U_target(t): return qt.qeye(2)*(-1j*H0*t).expm()
def process(rho, tlist):
    res = qt.mesolve(H0, rho, tlist)
    return [transform_state_unitary(rho, U_target(t), inverse = True) for rho,t in zip(res.states,tlist)]
tlist = np.linspace(0,np.pi/2,10)
F_av = average_fidelity(process, subspace_basis, tlist)
print(F_av,'\n')
