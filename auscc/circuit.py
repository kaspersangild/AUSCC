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
        expr = sp.Add(*[quantization_form(term,initial_expand= False) for term in expr.args])
    if expr.func == sp.Mul:
        expr = sp.Mul(*[quantization_form(factor,initial_expand= False) for factor in expr.args])
    if expr.func == sp.cos:
        expr = sp.expand_trig(expr)
    if expr.func == sp.sin:
        expr = sp.expand_trig(expr)
    if initial_expand: # Used to cancel out terms
        expr = sp.expand(expr)
    return expr



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


class circuit:
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

    def C_mat(self):
        N = max(max([b.start, b.end]) for b in self.branches)
        x = list(sp.symbols('x1:{}'.format(N+1))) # This may be modified when incorporating external controls...
        Vg = list(set([b.bias_voltage for b in self.branches]+[sp.diff(b.bias_flux,self._t) for b in self.branches]))
        print(Vg)
        node_voltages = [0] # Ground node
        if self.V == None:
            self.V = sp.eye(N)
        for n,xn in enumerate(self.V.inv()*sp.Matrix(x)):
            node_voltages.append(xn)
        C_mat11 = sp.zeros(N)
        C_mat12 = sp.zeros(N, len(Vg))
        for b in self.branches:
            b_voltage = node_voltages[b.end]-node_voltages[b.start]+b.bias_voltage+sp.diff(b.bias_flux, self._t)
            if b.type == 'Capacitor':
                v1 = sp.Matrix([b_voltage.coeff(xn) for xn in x]) # Vector with coefficients such that v.T*x_vec = branch flux
                v2 = sp.Matrix([b_voltage.coeff(Vgn) for Vgn in Vg])
                C_mat11 += b.symbol*v1*v1.T
                C_mat12 += b.symbol*v1*v2.T
        qg = C_mat12*sp.Matrix(Vg)
        return C_mat11, qg

    def kinetic(self, ignoreable_coordinates = []):
        N = max(max([b.start, b.end]) for b in self.branches)
        p = list(sp.symbols('p1:{}'.format(N+1)))
        C_mat = self.C_mat()
        K = (sp.Matrix(p).T*C_mat.inv()*sp.Matrix(p))[0,0]/2
        for cord in sorted(ignoreable_coordinates, reverse = True):
            if cord == 0:
                print('NB! Zeroth coordinate is ground and is automatically removed.')
            else:
                cord -= 1
                K = K.subs(p[cord], 0)
                del p[cord]
        return K,p

    def potential(self,ignoreable_coordinates = []):
        tp = sp.Symbol('t\'') # Integration variable
        N = max(max([b.start, b.end]) for b in self.branches)
        x = list(sp.symbols('x1:{}'.format(N+1)))# This may be modified when incorporating external controls...
        controls = [b.bias_voltage for b in self.branches if not b.bias_voltage == 0]\
                    +[b.bias_flux for b in self.branches if not b.bias_flux == 0]
        node_fluxes = [0] # Ground node
        if self.V == None:
            self.V = sp.eye(N)
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
                U -= b.symbol*sp.cos(b_flux)
        for cord in sorted(ignoreable_coordinates, reverse = True):
            if cord == 0:
                print('NB! Zeroth coordinate is ground and is automatically removed.')
            else:
                cord -= 1
                U = U.subs(x[cord-1], 0)
                del x[cord]
        U = quantization_form(U)
        return U,x

    def quantize(self, dims = [], taylor_order = 4, x0 = None, ignoreable_coordinates = []):
        """Quantizes the circuit.

        Parameters
        ----------
        dims : list
            List of integers denoting the desired dimension of each mode.
        taylor_order : Int
            The order to which the taylor expansion of the potential will be carried out
        x0 : list
            list (or numpy array) of floats. The point around which the taylour expansion of the potential is carried out
        ignoreable_coordinates : list
            List of integers. Specifies coordinates to be omitted during quantization. Typically CM-like coordinates. These coordinates are then simply set to zero.

        Returns
        -------
        symopgen
            symopgen which can generate the circuit Hamiltonian.

        """
        K,p = self.kinetic(ignoreable_coordinates = ignoreable_coordinates)
        U,x = self.potential(ignoreable_coordinates = ignoreable_coordinates)
        return au.quantize(K, U, p, x, dims = dims, taylor_order = taylor_order, x0 = x0)

    def __str__(self):
        N = max(max([b.start, b.end]) for b in self.branches)
        out = 'Circuit with '+str(N)+' node(s)\nBranches:\n'
        for branch in self.branches:
            out += branch.__str__()+'\n'
        return out

    def __init__(self, V = None):
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
        self.V = V
