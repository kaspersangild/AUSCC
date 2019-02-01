import qutip as qt
import numpy as np
import scipy as sp
import sympy as sym
from auscc.operator_generation import operator_generator

norm_tol = 1e-12
class subspace:
    def project(self, op):
        return self.subspace_projector*op*self.subspace_projector

    def diagonal(self, op):
        return np.sum([p*op*p for p in self.state_projectors])

    def __init__(self, states, diagonal_weight = 1., decoupled_weight = 1., degenerate_weight = 0):
        assert not (degenerate_weight>0 and diagonal_weight>0)
        self.weight = 1
        self.diagonal_weight = diagonal_weight
        self.decoupled_weight = decoupled_weight
        self.degenerate_weight = degenerate_weight
        self.states = states
        self.state_projectors = []
        for psi in states:
            assert psi.isket
            assert (psi.norm() - 1)**2 < norm_tol, 'Target eigenstates must be normalized'
            self.state_projectors.append(psi*psi.dag())
        self.subspace_projector = np.sum(self.state_projectors)
        self.dim = len(states)


def _diagonal_cost(op_proj, sub, norm_factor):
    return sub.diagonal_weight*((sub.diagonal(op_proj)-op_proj)**2).tr() / norm_factor

def _decoupling_cost(op, op_proj, sub, norm_factor):
    ## TO DO
    # This contribution is not good... Gotta do something where <psi_1|H|psi_2>/dE -> 0, where psi1 is eig in subspace and psi2 is is eig not in subspace. Don't know if this can be done without diagonalizing entire hamiltonian...
    return sub.decoupled_weight*((sub.subspace_projector*op + op*sub.subspace_projector - 2*op_proj)**2).tr() / norm_factor

def _cost(params, op_gen, subspaces):
    cost = 0.
    op = op_gen(params)
    full_norm = (op**2).tr()
    for sub in subspaces:
        op_proj = sub.project(op)
        cnst_offset = op_proj.tr() / sub.dim
        if (not sub.diagonal_weight == 0) or (not sub.degenerate_weight == 0):
            norm_factor = ((op_proj-cnst_offset*sub.subspace_projector)**2).tr()
        if not sub.diagonal_weight == 0:
            if norm_factor > 0:
                cost +=_diagonal_cost(op_proj-cnst_offset, sub, norm_factor)
        if not sub.decoupled_weight == 0:
            cost += _decoupling_cost(op, op_proj, sub, full_norm)
        if not sub.degenerate_weight == 0:
            cost += sub.degenerate_weight*norm_factor/full_norm
    return cost


def param_optim(op_gen, x0, subspaces = [], min_gaps = [], non_degen_states = [], method = None, bounds = None, constraints = (), tol = None):
    norm_tol = 1e-12
    if isinstance(subspaces, subspace):
        subspaces = [subspaces]
    args = (op_gen, subspaces)
    return sp.optimize.minimize(_cost, x0, args, method = method, bounds = bounds, constraints = constraints, tol = tol)


if __name__ == '__main__':
    x, z = sym.symbols('x z')
    coeffs = [sym.lambdify([x, z], x), sym.lambdify([x, z], z), sym.lambdify([x, z], 1)]
    ops = [qt.sigmax(), qt.sigmaz(), qt.qeye(2)]
    state0 = qt.ket('0')
    state1 = qt.ket('1')
    op_gen = operator_generator(coeffs, ops)
    target_eig_state = [(state0+state1).unit(), (state0-state1).unit()]
    subs = subspace(target_eig_state)
    res = param_optim(op_gen, np.array([1, 1]), subs)
    print(res)
    omega1, omega2, jx = sym.symbols('omega1 omega2 jx')
    coeffs = [  sym.lambdify((omega1,omega2,jx),omega1/2),
                sym.lambdify((omega1,omega2,jx), omega2/2),
                sym.lambdify((omega1,omega2,jx), jx)]
    ops = [     qt.tensor(qt.sigmaz(), qt.qeye(2)),
                qt.tensor(qt.qeye(2), qt.sigmaz()),
                qt.tensor(qt.sigmax(), qt.sigmax())]
    op_gen = operator_generator(coeffs, ops)
    states1 =            [   qt.ket('00'),
                            qt.ket('11')]
    states2 =            [  (qt.ket('10')+qt.ket('01')).unit(),
                            (qt.ket('10')-qt.ket('01')).unit()]
    subs = [subspace(states1, diagonal_weight = 0, decoupled_weight = 1.),
            subspace(states2, diagonal_weight = 0, decoupled_weight = 1., degenerate_weight = -1)]
    res = param_optim(op_gen, np.array([1,1,0.1]), subs)
    print(res)

    # Transmon C = 1. Driving from 0-1 x = [EJ, A]
    b = qt.destroy(3)
    coeffs = [lambda alpha: -alpha/2, lambda alpha: 1.,]
    ops = [b.dag()*b.dag()*b*b, b.dag()+b]
    op_gen = operator_generator(coeffs,ops)
    states = [  (qt.ket('0', 3)+qt.ket('1', 3)).unit(),
                (qt.ket('0', 3)-qt.ket('1', 3)).unit() ]
    subs = subspace(states, diagonal_weight = 1., decoupled_weight = 1., degenerate_weight = -0.1)
    res = param_optim(op_gen, [0.1], subs)
    print(res)
    op = op_gen(res.x)
    E, e_states = op.eigenstates()
    print(E)
    print(e_states)
