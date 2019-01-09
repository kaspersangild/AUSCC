from auscc.fidelity import average_fidelity, entanglement_fidelity, leakage
import qutip as qt
import numpy as np

if __name__ == '__main__':
    # One qubit simple test unitary basis
    basis = [qt.ket([0],[2]), qt.ket([1],[2])]
    U_target = -1j*qt.sigmax()
    H = qt.sigmax()
    def process(rho, tlist, e_ops):
        res = qt.mesolve(H, rho, tlist, e_ops = U_target*e_ops*U_target.dag())
        return res.expect[0]

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
    def process(rho, tlist, e_ops):
        res = qt.mesolve(H, rho, tlist,c_ops = [0.1*qt.destroy(3)] , e_ops = U_target*e_ops*U_target.dag())
        return res.expect[0]
    tlist = np.linspace(0,np.pi/2,10)
    F_av = average_fidelity(process, subspace_basis, tlist)
    print(F_av,'\n')
    # One qudit with four states [1,2] comp basis
    subspace_basis =  [qt.ket([1], 4), qt.ket([2], 4) ]
    P = projection_operator(subspace_basis, [2])
    U_target = P.dag()*(-1j*qt.sigmax())*P
    H = qt.create(4)+qt.destroy(4)+20*qt.ket([3],[4])*qt.bra([3],[4])+20*qt.ket([0],[4])*qt.bra([0],[4])
    def process(rho, tlist, e_ops):
        res = qt.mesolve(H, rho, tlist, e_ops = U_target*e_ops*U_target.dag())
        return res.expect[0]
    tlist = np.linspace(0,np.pi/(2*np.sqrt(2)),10)
    F_av = average_fidelity(process, subspace_basis, tlist)
    print(F_av,'\n')
    # One qubit and one qutrit + time dependent hamiltonian (CZ gate)
    # subspace_basis =  [qt.ket(seq, [2,3]) for seq in qt.state_number_enumerate([2,2])]
    # P = projection_operator(subspace_basis, [2,2])
    # U_target_I = (-qt.tensor(qt.sigmaz(),qt.sigmaz())+qt.tensor(qt.sigmaz(),qt.qeye(2))+qt.tensor(qt.qeye(2),qt.sigmaz())+1)/2
    # U_target_I =
    # args = {'omega':100, 'A' : np.sqrt(2)}
    # H0 = args['omega']*(qt.tensor(qt.num(2),qt.num(3)) + qt.tensor(qt.qeye(2),qt.num(3)**2))
    # H0_comp = args['omega']*(qt.tensor(qt.num(2),qt.num(2)) + qt.tensor(qt.qeye(2), qt.num(2)**2))
    # tlist = np.linspace(0,np.pi,10)
    # U_target_S = []
    # for t in tlist:
    #     U_target_S.append((-1j*t*H0_comp).expm()*U_target_I*(1j*0*H0_comp).expm())
    # H1 = [qt.tensor(qt.qeye(2),qt.create(3)+qt.destroy(3)),'A*cos(4*omega*t)']
    # H = [H0,H1]
    # F_av = average_fidelity(U_target_S, H, tlist, c_ops = [], args = args,basis = 'unitary',progress_bar = True)
    # print(F_av)
