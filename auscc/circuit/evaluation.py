from numpy import array, float64
from numpy.linalg import inv
from sympy import sympify, Add, Mul, Matrix
from auscc.circuit.operators import build_operator
from qutip import qzero

def seperate_expression(expr, op_syms, circ_syms, ctrl_syms):
    if expr.func == Add:
        out = []
        for term in expr.args:
            out += seperate_expression(term, op_syms, circ_syms, ctrl_syms)
        return out
    else:
        op_factor = sympify(1.)
        cnst_factor = sympify(1.)
        ctrl_factor = sympify(1.)
        if expr.func == Mul:
            factors = expr.args
        else:
            factors = (expr,)
        for fac in factors:
            if all(S in circ_syms for S in fac.free_symbols):
                cnst_factor *= fac
            elif all(S in ctrl_syms for S in fac.free_symbols):
                ctrl_factor *= fac
            elif all(S in op_syms for S in fac.free_symbols):
                op_factor *= fac
            else:
                raise ValueError("Could not handle expression {}".format(str(expr)))
        return [(cnst_factor.evalf(), op_factor.evalf(), ctrl_factor.evalf())]



class Circuit_evaluator:
    """This class is responsible for the evaluation of the circuit. Once all operators have been build, this is the object that is evaluates all coefficients and returns the hamiltonian on the correct form. It also holds the quantization form of the potential. This is where the potential has been seperated into terms that can be seperated into factors only containing either a circuit parameters, control paremeters or symbols representing operators. These operator symbols serves as keys for the operator library where their qutip counterparts are stored.

    Parameters
    ----------
    ops_lib : type
        Description of parameter `ops_lib`.

    Attributes
    ----------
    ops_lib

    """

    def eval_using_subs(self, ops_lib, circ_subs, ctrl_subs = []):
        circ_subs = dict(circ_subs)
        ctrl_subs = dict(ctrl_subs)
        for key in self.circ_syms:
            if not key in circ_subs.keys():
                circ_subs[key] = 0.
        for key in self.ctrl_syms:
            if not key in ctrl_subs.keys():
                ctrl_subs[key] = 0.
        C_inv = array(self.C_mat.subs(circ_subs)).astype(float64)
        C_inv = Matrix(inv(C_inv))
        K = (self.p.T*C_inv*self.p/2 + self.pg.T*C_inv*self.p)[0,0]
        K_terms = seperate_expression(K, self.op_syms, self.circ_syms, self.ctrl_syms)
        H0 = 0
        H1 = []
        for (circ_fac, op_fac, ctrl_fac) in list(K_terms)+list(self.U_terms):
            ctrl_fac = sympify(ctrl_fac).subs(ctrl_subs)
            if ctrl_fac.free_symbols:
                H1.append([float(circ_fac.subs(circ_subs))*ops_lib(op_fac), str(ctrl_fac)])
            else:
                H0 += float(circ_fac.subs(circ_subs)*ctrl_fac)*ops_lib(op_fac)
        if H1:
            return [H0]+H1
        else:
            return H0


    def __init__(self, x_syms, p_syms, U, C_mat, qg, circ_syms, ctrl_syms):
        self.op_syms = list(x_syms)+list(p_syms)
        self.circ_syms = circ_syms
        self.ctrl_syms = ctrl_syms
        self.U_terms = seperate_expression(U, self.op_syms, self.circ_syms, self.ctrl_syms)
        self.C_mat = C_mat
        self.pg = Matrix(qg)
        self.p = Matrix(p_syms)
