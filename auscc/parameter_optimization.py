import qutip as qt
import numpy as np
import scipy as sp
import sympy as sym
from auscc.operator_generation import operator_generator

def _target_eigen_states_cost(op_proj, projectors):
    D = np.sum([p*op_proj*p for p in projectors])
    return ((D-op_proj)**2).tr()

def _decoupling_cost(op, op_proj, P):
    return ((P*op+op*P-2*op_proj)**2).tr()

def _minimize_energy_gap_cost(op, state1, state2):
    return (qt.expect(op,state1)-qt.expect(op,state2))**2

def _non_degenerate_cost(op, state1, state2):
    return (qt.expect(op,state1)-qt.expect(op,state2))**-2

def _cost(params, op_gen, target_eigen_states, projectors, P, min_gaps, non_degen_states):
    cost = 0.
    op = op_gen(params)
    norm_factor = (op**2).tr()
    op_proj = P*op*P
    if target_eig_state:
        cost += _target_eigen_states_cost(op_proj, projectors)/norm_factor
        cost += _decoupling_cost(op, op_proj, P)/norm_factor
    for gap in min_gaps:
        w = gap[0]
        state1 = gap[1]
        state2 = gap[2]
        cost += w*_minimize_energy_gap_cost(op,state1,state2)/norm_factor
    for wstates in non_degen_states:
        w = wstates[0]
        state1 = wstates[1]
        state2 = wstates[2]
        cost += w*_non_degenerate_cost(op, state1, state2)*norm_factor

    return cost


def param_optim(op_gen, x0, target_eigen_states = [], min_gaps = [], non_degen_states = [], method = None, bounds = None, constraints = ()):
    costs = []
    projectors = []
    P = 0
    norm_tol = 1e-12
    if target_eigen_states:
        for psi in target_eigen_states:
            assert psi.isket
            assert (psi.norm() - 1)**2 < norm_tol, 'Target eigenstates must be normalized'
            projectors.append(psi*psi.dag())
        P = np.sum(projectors)
    args = (op_gen, target_eigen_states, projectors, P, min_gaps, non_degen_states)
    return sp.optimize.minimize(_cost, x0, args, method = method, bounds=bounds, constraints=constraints)


if __name__ == '__main__':
    x,z = sym.symbols('x z')
    coeffs = [sym.lambdify((x,z), x), sym.lambdify((x,z), z)]
    ops = [qt.sigmax(), qt.sigmaz()]
    state0 = qt.ket('0')
    state1 = qt.ket('1')
    op_gen = operator_generator(coeffs, ops)
    target_eig_state = [(state0+state1).unit(), (state0-state1).unit()]
    res = param_optim(op_gen, np.array([1, 1]), target_eig_state)
    print(res.x)
    print(op_gen(res.x))
    omega1, omega2, jx = sym.symbols('omega1 omega2 jx')
    coeffs = [  sym.lambdify((omega1,omega2,jx),omega1/2),
                sym.lambdify((omega1,omega2,jx), omega2/2),
                sym.lambdify((omega1,omega2,jx), jx)]
    ops = [     qt.tensor(qt.sigmaz(), qt.qeye(2)),
                qt.tensor(qt.qeye(2), qt.sigmaz()),
                qt.tensor(qt.sigmax(), qt.sigmax())]
    op_gen = operator_generator(coeffs, ops)
    states =            [   qt.ket('00'),
                            qt.ket('11'),
                            (qt.ket('10')+qt.ket('01')).unit(),
                            (qt.ket('10')-qt.ket('01')).unit()]
    non_degen = [[1e-5, states[2], states[3]]]
    res = param_optim(op_gen, np.array([10,10,0.1]), states, non_degen_states = non_degen)
    print(res)

    # Transmon C = 1. Driving from 0-1 x = [EJ, A]
    b = qt.destroy(3)
    coeffs = [lambda alpha: -alpha/2, lambda alpha: 1.,]
    ops = [b.dag()*b.dag()*b*b, b.dag()+b]
    op_gen = operator_generator(coeffs,ops)
    states = [  (qt.ket('0', 3)+qt.ket('1', 3)).unit(),
                (qt.ket('0', 3)-qt.ket('1', 3)).unit() ]
    non_degen = [[1e-4, states[0], states[1]]]
    res = param_optim(op_gen, [0.1], target_eigen_states = states, non_degen_states = non_degen)
    print(res)
