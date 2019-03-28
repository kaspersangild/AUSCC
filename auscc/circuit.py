from auscc.taylor import taylor_sympy
import auscc as au
import sympy as sp
import numpy as np
import qutip as qt



class branch:
    def __str__(self):
        return '{0:20}{1:10} from {2} to {3}'.format(self.type, str(self.symbol), str(self.start), str(self.end))

    def __init__(self, start, end, type, symbol):
        self.start = start
        self.end = end
        self.symbol = symbol
        self.type = type

class circuit:
    """Class for a superconducting circuit.
    Parameters
    ----------
    V : sympy matrix
        Transformation matrix used such that x = V*flux_nodes_vector, where x is the desired coordinates. Must be invertible.
    """

    def C_mat(self):
        N = max(max([b.start, b.end]) for b in self.branches)
        x = list(sp.symbols('x1:{}'.format(N+1)))# This may be modified when incorporating external controls...
        p = list(sp.symbols('p1:{}'.format(N+1)))
        node_fluxes = [0] # Ground node
        x = list(sp.symbols('x1:{}'.format(N+1)))
        C_mat = sp.zeros(N)
        for b in self.branches:
            b_flux = node_fluxes[b.end]-node_fluxes[b.start] # Her kan man ændre hvis man vil have externe fluxer og lignende ind.
            if b.type == 'Capacitor':
                v = sp.Matrix([b_flux.coeff(xn) for xn in x]) # Vector with coefficients such that v.T*x_vec = branch flux
                C_mat += b.symbol*v*v.T
        return C_mat

    def potential(self):
        N = max(max([b.start, b.end]) for b in self.branches)
        x = list(sp.symbols('x1:{}'.format(N+1)))# This may be modified when incorporating external controls...
        p = list(sp.symbols('p1:{}'.format(N+1)))
        node_fluxes = [0] # Ground node
        U = 0
        for b in self.branches:
            b_flux = node_fluxes[b.end]-node_fluxes[b.start] # Her kan man ændre hvis man vil have externe fluxer og lignende ind.
            if b.type == 'Inductor':
                U += (b_flux)**2/(2*b.symbol)
            elif b.type == 'Josephson junction':
                U -= b.symbol*sp.cos(b_flux)
        return U

    def add_branch(self, start, end, type, symbol):
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
            'Josephson junction' : 'Josephson junction',
            'Josephson Junction' : 'Josephson junction',
            'J' : 'Josephson junction',
            'JJ' : 'Josephson junction'
        }
        type = switcher[type]
        self.branches.append(branch(start, end, type, symbol))

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
        N = max(max([b.start, b.end]) for b in self.branches)
        x = list(sp.symbols('x1:{}'.format(N+1)))# This may be modified when incorporating external controls...
        p = list(sp.symbols('p1:{}'.format(N+1)))
        node_fluxes = [0] # Ground node
        if self.V == None:
            self.V = sp.eye(N)
        for n,xn in enumerate(self.V.inv()*sp.Matrix(x)):
            node_fluxes.append(xn)
        C_mat = sp.zeros(N)
        U = 0
        for b in self.branches:
            b_flux = node_fluxes[b.end]-node_fluxes[b.start] # Her kan man ændre hvis man vil have externe fluxer og lignende ind.
            if b.type == 'Capacitor':
                v = sp.Matrix([b_flux.coeff(xn) for xn in x]) # Vector with coefficients such that v.T*x_vec = branch flux
                C_mat += b.symbol*v*v.T
            elif b.type == 'Inductor':
                U += (b_flux)**2/(2*b.symbol)
            elif b.type == 'Josephson junction':
                U -= b.symbol*sp.cos(b_flux)
        K = (sp.Matrix(p).T*C_mat.inv()*sp.Matrix(p))[0,0]/2
        for cord in sorted(ignoreable_coordinates, reverse = True):
            K = K.subs(p[cord], 0)
            U = U.subs(x[cord], 0)
            del p[cord]
            del x[cord]
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
        self.V = V
