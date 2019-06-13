from auscc.taylor import taylor_sympy
import auscc as au
import sympy as sp
import numpy as np
import qutip as qt

def quantization_form(expr, initial_expand = True):
    """Brings a sympy expression on a form that the quantizers can understand.

    Parameters
    ----------
    expr : sympy expression
        The expression. Typically a symbolic potential or kinetic term
    Returns
    -------
    sympy expr
        The rewritten expression.

    """
    if initial_expand:
        expr = sp.expand(expr)
    if expr.func == sp.Add:
        expr = sp.Add(*[quantization_form(term,initial_expand = False) for term in expr.args])
    if expr.func == sp.Mul:
        expr = sp.Mul(*[quantization_form(factor,initial_expand = False) for factor in expr.args])
    if expr.func == sp.cos:
        expr = sp.expand_trig(expr)
    if expr.func == sp.sin:
        expr = sp.expand_trig(expr)
    if initial_expand: # Used to cancel out terms
        expr = sp.expand(expr)
    return expr

def C_mat_coeffs(C_mat):
    return np.flatten(np.linalg.inv(C_mat)/2)

def dft_quantization_ops(dims):
    p = []
    for ind in range(len(dims)):
        p.append(qt.tensor([qt.charge((d-1)/2) if i == ind else qt.qeye(d) for i,d in enumerate(dims)]))
    return p


class branch:
    def __str__(self):
        return '{0:20}{1:10} from {2} to {3}'.format(self.type, str(self.symbol), str(self.start), str(self.end))

    def __init__(self, start, end, type, symbol, bias_voltage, bias_flux):
        self.start = start
        self.end = end
        self.symbol = symbol
        self.type = type
        self.bias_voltage = bias_voltage
        self.bias_flux = bias_flux


class Circuit:
    """Class for a superconducting circuit.
    Parameters
    ----------
    V : sympy matrix
        Transformation matrix used such that x = V*flux_nodes_vector, where x is the desired coordinates. Must be invertible.
    """

    def add_branch(self, start, end, type, symbol,bias_voltage = 0, bias_flux = 0):
        """Adds a branch to the circuit

        Parameters
        ----------
        start : int
            The starting flux node.
        end : int
            the ending flux node.
        type : string
            Specifies the branch element. Currently 'Inductor', 'Capacitor', and 'Josephson junction' are available.
        symbol : sympy symbol
            The parameter of the circuit element, i.g. C, for a capacitor where C is some sympy symbol representing the capacitance.
        """

        switcher = {
            'Inductor' : 'Inductor',
            'L' : 'Inductor',
            'Capacitor' : 'Capacitor',
            'C' : 'Capacitor',
            'capacitor': 'Capacitor',
            'Josephson junction' : 'Josephson junction',
            'Josephson Junction' : 'Josephson junction',
            'J' : 'Josephson junction',
            'JJ' : 'Josephson junction',
            'JJunction' : 'Josephson junction'
        }
        type = switcher[type]
        self.branches.append(branch(start, end, type, symbol, bias_voltage, bias_flux))

    def circuit_symbols(self):
        circsyms = []
        for b in self.branches:
            if not b.symbol in circsyms:
                circsyms.append(b.symbol)
        return circsyms

    def control_symbols(self):
        controlsyms = []
        for b in self.branches:
            if not b.bias_voltage in controlsyms:
                controlsyms.append(b.bias_voltage)
            if not b.bias_flux in controlsyms:
                controlsyms.append(b.bias_flux)
        controlsyms.remove(0)
        return controlsyms

    def C_mat(self):
        N = max(max([b.start, b.end]) for b in self.branches)+1 # Number of nodes
        x = list(sp.symbols('x0:{}'.format(N))) # This may be modified when incorporating external controls...
        Vg = list(set([b.bias_voltage for b in self.branches]+[sp.diff(b.bias_flux,self._t) for b in self.branches]))
        node_voltages = [] # Ground node
        if self.V == None:
            self.V = sp.eye(N)
        for n,yn in enumerate(self.V.inv()*sp.Matrix(x)):
            node_voltages.append(yn)
        C_mat11 = sp.zeros(N-len(self.ignoreable_coordinates))
        C_mat12 = sp.zeros(N-len(self.ignoreable_coordinates), len(Vg))
        for cord in sorted(self.ignoreable_coordinates, reverse = True):
                del x[cord]
        for b in self.branches:
            b_voltage = node_voltages[b.end]-node_voltages[b.start]+b.bias_voltage+sp.diff(b.bias_flux, self._t)
            if b.type == 'Capacitor':
                v1 = sp.Matrix([b_voltage.coeff(xn) for xn in x]) # Vector with coefficients such that v.T*x_vec = branch flux
                v2 = sp.Matrix([b_voltage.coeff(Vgn) for Vgn in Vg])
                C_mat11 += b.symbol*v1*v1.T
                C_mat12 += b.symbol*v1*v2.T
        qg = C_mat12*sp.Matrix(Vg)
        return C_mat11, qg

    def kinetic(self,subs = []):
        N = max(max([b.start, b.end]) for b in self.branches)+1
        p = list(sp.symbols('p0:{}'.format(N)))
        for cord in sorted(self.ignoreable_coordinates, reverse = True):
            del p[cord]
        C_mat, qg = self.C_mat()
        K = (sp.Matrix(p).T*C_mat.inv()*sp.Matrix(p))[0,0]/2
        return p,K.subs(subs)

    def potential(self,subs = []):
        tp = sp.Symbol('t\'') # Integration variable
        N = max(max([b.start, b.end]) for b in self.branches)
        x = list(sp.symbols('x0:{}'.format(N+1)))# This may be modified when incorporating external controls...
        controls = [b.bias_voltage for b in self.branches if not b.bias_voltage == 0]\
                    +[b.bias_flux for b in self.branches if not b.bias_flux == 0]
        node_fluxes = []
        if self.V == None:
            self.V = sp.eye(N+1)
        for n,xn in enumerate(self.V.inv()*sp.Matrix(x)):
            node_fluxes.append(xn)
        U = 0
        for b in self.branches:
            if b.bias_voltage == 0:
                bias_V_flux = 0
            else:
                bias_V_flux = sp.integrate(b.bias_voltage.subs(self._t,tp),(tp, -sp.oo, self._t))
            b_flux = node_fluxes[b.end]-node_fluxes[b.start]+b.bias_flux+bias_V_flux
            if b.type == 'Inductor':
                U += (b_flux)**2/(2*b.symbol)
            elif b.type == 'Josephson junction':
                U += b.symbol*(1-sp.cos(b_flux))
        for cord in sorted(self.ignoreable_coordinates, reverse = True):
                U = U.subs(x[cord], 0)
                del x[cord]
        U = quantization_form(U)
        return x,U.subs(subs)

    def quantize(self, dims = [], taylor_order = 4, x0 = None):
        """Quantizes the circuit.

        Parameters
        ----------
        dims : list
            List of integers denoting the desired dimension of each mode.
        taylor_order : Int
            The order to which the taylor expansion of the potential will be carried out
        x0 : list
            list (or numpy array) of floats. The point around which the taylour expansion of the potential is carried out

        Returns
        -------
        symopgen
            symopgen which can generate the circuit Hamiltonian.

        """
        p,K = self.kinetic()
        x,U = self.potential()
        return au.quantize_SHO(K, U, p, x, dims = dims, taylor_order = taylor_order, x0 = x0)

    def build_OG(self, method = 'dft', dims = []):
        # Kinetic stuff
        C_mat, qg = self.C_mat()
        x, U = self.potential()
        vars_sym = list((set(C_mat.free_symbols) | set(U.free_symbols)) - set(x))
        if method == 'dft':
            self.OG = build_OG_dft()
        return 0

    def __str__(self):
        N = max(max([b.start, b.end]) for b in self.branches)
        out = 'Circuit with '+str(N)+' node(s)\nBranches:\n'
        for branch in self.branches:
            out += branch.__str__()+'\n'
        return out

    def __init__(self):
        """Short summary.

        Parameters
        ----------
        V : type
            Description of parameter `V`.

        Returns
        -------
        type
            Description of returned object.

        """
        self.branches = []
        self._t = sp.Symbol('t', real = True)
        self.V = None
        self.ignoreable_coordinates = [0]
