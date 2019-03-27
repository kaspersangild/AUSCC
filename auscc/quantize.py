import auscc as au
import sympy as sp
import numpy as np
import qutip as qt

def quantize(K, U, p, x, dims = [], taylor_order = 4, x0 = None):
    """A function that can quantize a sympy hamiltonian. Currently only in SHO/fock basis

    Parameters
    ----------
    K : sympy expression
        Sympy expression for the kinetic term
    U : sympy expression
        sympy expression for the potential
    p : iterable of sympy symbols
        Symbols representing the canonical momenta of the system
    x : iterable of sympy symbols
        Symbols representing the canonical position variables of the system
    dims : list of integers
        Specifies the desired dimension of each mode. Must be same length as x and p
    taylor_order : integer
        Specifies the order to which the taylor expansion will be carried out if the quantization method utilises, such an expansion.
    x0 : list of floats
        The point around which the taylor expansion of the potential wil be carried out.

    Returns
    -------
    operator_generator
        An operator generator that when called as instance(*params) return the hamiltonian for the system with given params. A parameter is defined as a symbol that appears in K or U but not in x or p.

    """
    # Preliminaries
    if not dims:
        dims = len(x)*[4]
    if x0 == None:
        x0 = np.zeros(len(x))

    # Taylor expansion
    T_U = au.taylor.taylor_sympy(f=U, x=x, x0=x0, N=taylor_order)
    T_K = au.taylor.taylor_sympy(f=K, x=p, x0=np.zeros(len(p)), N=2)

    # Effective mass and "spring constant"
    m = len(x)*[0]
    k = len(x)*[0]
    for coeff,powers in T_K:
        inds = np.nonzero(np.array(powers) == 2)[0]
        if len(inds) == 1:
            m[inds[0]] = sp.simplify(1/(2*coeff))
    for coeff,powers in T_U:
        inds = np.nonzero(np.array(powers) == 2)[0]
        if len(inds) == 1:
            k[inds[0]] = sp.simplify(2*coeff)

    # Constructing position and momentum operator
    x_ops = []
    p_ops = []
    padded_dims = [d+int(np.floor(taylor_order/2)) for d in dims]
    for j,kj,mj,d in zip(range(len(k)),k, m, padded_dims):
        op = [qt.qeye(d) for d in padded_dims]
        op[j] = 1j*(qt.create(d)-qt.destroy(d))/np.sqrt(2)
        p_ops.append(qt.tensor(op))
        op[j] = (qt.create(d)+qt.destroy(d))/np.sqrt(2)
        x_ops.append(qt.tensor(op))
    terms = []

    # Calculating qutip operator and symbolic coefficient for each term
    P = 0 # This will be the projection operator that projects onto the subspace defined by dims
    for state in qt.state_number_enumerate(dims):
        P += qt.ket(state, dims)*qt.bra(state, padded_dims)
    for coeff, powers in T_K:
        op = qt.qeye(padded_dims)
        for kj,mj,p_op,pow in zip(k,m,p_ops,powers):
            if pow > 0:
                coeff *= (kj*mj)**(pow/4)
                op *= p_op**pow
        terms.append((coeff, P*op*P.dag()))
    for coeff, powers in T_U:
        op = qt.qeye(padded_dims)
        for kj,mj,x_op,pow in zip(k,m,x_ops,powers):
            if pow > 0:
                coeff *= (kj*mj)**(-pow/4)
                op *= x_op**pow
        terms.append((sp.simplify(coeff), P*op*P.dag()))

    # Constructing opgen instance
    return au.symopgen(sym_terms=terms)
