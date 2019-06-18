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


class Branch:
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
    ####################################################################################################################
    # Circuit building
    ####################################################################################################################
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
        N_before = 0 if len(self.branches) == 0 else self.number_of_nodes()
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
        self.branches.append(Branch(start, end, type, symbol, bias_voltage, bias_flux))
        if not N_before == self.number_of_nodes():
            self.reset_coords()

    ##################################################################################################################### Coordinates
    ####################################################################################################################
    def number_of_nodes(self):
        return max(max(b.start, b.end)  for b in self.branches)+1

    def number_of_coords(self):
        return self.number_of_nodes() - len(self.ignored_coords)

    def ignore_coords(self,*args):
        new_coords = list(self.ignored_coords)
        for ind in args:
            assert isinstance(ind, int), 'all inputs must be integers'
            if not ind in new_coords:
                new_coords.append(ind)
        V = self._V # We don't want to reset the transformation matrix
        self.reset_coords()
        self.ignored_coords = tuple(sorted(new_coords))
        self.V = V

    def use_coords(self,*args):
        new_coords = list(self.ignored_coords)
        for ind in args:
            assert isinstance(ind, int), 'all inputs must be integers'
            if ind in new_coords:
                new_coords.remove(ind)
        V = self._V
        self.reset_coords()
        self.ignored_coords = tuple(sorted(new_coords))
        self.V = V

    def include_ignored_coords(self, vec):
        N = self.number_of_nodes()
        out = sp.zeros(N, 1)
        used_coords = [i for i in range(N) if i not in self.ignored_coords]
        for vec_ind, ind in zip(vec, used_coords):
            out[ind] = vec_ind
        return out

    def transform_coords(self, *args): # This method can definitely be made more flexible and user friendly.
        N = self.number_of_nodes()
        assert len(args) == N, 'Inputs must match number of nodes (including ground!)'
        assert all([len(v) == N for v in args]), 'Length of all inputs must match number of nodes (including ground!)'
        self.reset_coords()
        self._V = sp.Matrix([v for v in args])

    def get_V_mat(self):
        if self._V == None:
            N = self.number_of_nodes()
            self._V = sp.eye(N)
        return self._V

    def get_coord_subscripts(self, return_only_used = True):
        if self._coord_subscripts == None:
            N = self.number_of_nodes()
            self._coord_subscripts = tuple(str(i) for i in range(N))
        if not return_only_used:
            return self._coord_subscripts
        else:
            return tuple(subscript_i for i,subscript_i in enumerate(self._coord_subscripts) if not i in self.ignored_coords)

    def get_x_vec(self):
        if self._x == None:
            self._x = sp.Matrix(sp.symbols([r'x_{'+subs+'}' for subs in self.get_coord_subscripts(return_only_used=True)]))
        return self._x

    def get_v_vec(self):
        if self._v == None:
            self._v = sp.Matrix(sp.symbols([r'v_{'+subs+'}' for subs in self.get_coord_subscripts(return_only_used=True)]))
        return self._v

    def get_p_vec(self):
        if self._p == None:
            self._p = sp.Matrix(sp.symbols([r'p_{'+subs+'}' for subs in self.get_coord_subscripts(return_only_used=True)]))
        return self._p

    def get_node_x_vec(self):
        if self._node_x == None:
            Vinv = self.get_V_mat().inv()
            x = self.include_ignored_coords(self.get_x_vec())
            self._node_x = Vinv*x
        return self._node_x

    def get_node_v_vec(self):
        if self._node_v == None:
            Vinv = self.get_V_mat().inv()
            v = self.include_ignored_coords(self.get_v_vec())
            self._node_v = Vinv*v
        return self._node_v

    def reset_coords(self):
        self.ignored_coords = tuple()
        self._V = None
        self._x = None
        self._v = None
        self._p = None
        self._node_x = None
        self._node_v = None
        self._node_p = None
        self._coord_subscripts = None

    ####################################################################################################################
    # Symbolic manipulation
    ####################################################################################################################
    def circuit_symbols(self):
        circsyms = []
        for b in self.branches:
            for sym in b.symbol.free_symbols:
                if not sym in circsyms:
                    circsyms.append(sym)
        return sp.Matrix(circsyms)

    def control_symbols(self):
        controlsyms = []
        for b in self.branches:
            if not b.bias_voltage == 0:
                for sym in b.bias_voltage.free_symbols:
                    if not sym in controlsyms:
                        controlsyms.append(sym)
            if not b.bias_flux == 0:
                for sym in b.bias_flux.free_symbols:
                    if not sym in controlsyms:
                        controlsyms.append(sym)
        return sp.Matrix(controlsyms)

    def C_mat(self):
        v = self.get_v_vec()
        Vg = list(set([b.bias_voltage for b in self.branches]+[sp.diff(b.bias_flux,self._t) for b in self.branches]))
        node_voltages = self.get_node_v_vec()
        C_mat11 = sp.zeros(self.number_of_coords())
        C_mat12 = sp.zeros(self.number_of_coords(), len(Vg))
        for b in self.branches:
            b_voltage = node_voltages[b.end]-node_voltages[b.start]+b.bias_voltage+sp.diff(b.bias_flux, self._t)
            if b.type == 'Capacitor':
                c1 = sp.Matrix([b_voltage.coeff(vn) for vn in v]) # Vector with coefficients such that v1.T*v_vec = branch voltage
                c2 = sp.Matrix([b_voltage.coeff(Vgn) for Vgn in Vg])
                C_mat11 += b.symbol*c1*c1.T
                C_mat12 += b.symbol*c1*c2.T
        qg = C_mat12*sp.Matrix(Vg)
        return C_mat11, qg

    def kinetic(self,subs = []):
        p = self.get_p_vec()
        C_mat, qg = self.C_mat()
        K = (p.T*C_mat.inv()*p/2+p.T*qg)[0,0]
        return p,K.subs(subs)

    def potential(self,subs = []):
        tp = sp.Symbol('t\'') # Integration variable
        x = self.get_x_vec()
        node_fluxes = self.get_node_x_vec()
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
        U = quantization_form(U)
        return x,U.subs(subs)

    ####################################################################################################################
    # Quantization
    ####################################################################################################################
    def reset_quantization(self):
        self.quantization_settings = {}

    def set_quantization_method(self, method_list):
        N = self.number_of_cords
        if isinstance(method_list,str):
            method_list = [method_list]
        assert isinstance(method_list, list), 'Input must be string or list'
        if len(method_list) == 1:
            method_list = N*method_list
        assert len(method_list) == N, 'Invalid length of input'
        self.quantization_settings['method'] = method_list

    def set_p_ops(self, basis, dims):
        N = max(max(b.start, b.end)  for b in self.branches) - len(self.ignored_coords)
        assert len(dims) == N
        q_ops = []
        if isinstance(basis, Str):
            basis = N*[basis]
        # Building vector with q_n as n'th entry
        for n in range(N):
            op = [qt.qeye(d) for d in dims]
            if basis[n] == 'charge':
                op[n] = qt.charge((dims[n]-1)/2)
            q_ops.append(qt.tensor(op))
        q_ops = np.array(q_ops, dtype = 'O')
        # Building matrix with q_n*q_m as n,m entry
        q_ops_mat = np.empty(shape = (N,N), dtype = 'O')
        for n,qn in enumerate(q_ops):
            for m,qm in enumerate(q_ops):
                q_ops_mat[n,m] = q_ops
        self.q_ops_vec = q_ops
        self.q_ops_mat = q_ops_mat

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
        self.reset_coords()
        self.reset_quantization()
