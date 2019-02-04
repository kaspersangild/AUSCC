from auscc.taylor import taylor_sympy
import auscc as au
import sympy as sp
import numpy as np
import qutip as qt
class branch:
    def __str__(self):
        return '{0:20}{1:10} from {2} to {3}'.format(self.type_str, self.param_key, str(self.start), str(self.end))

    def __init__(self,start, end, param_key, branch_flux):
        self.start = start
        self.end = end
        self.param_key = param_key
        self.branch_flux = branch_flux
        self.type_str = 'Generic Branch'

class inductive_branch(branch):
    def __init__(self,start, end, param_key, branch_flux):
        branch.__init__(self,start, end, param_key, branch_flux)

class capacitative_branch(branch):
    def __init__(self,start, end, param_key, branch_flux):
        branch.__init__(self,start, end, param_key, branch_flux)

class inductor(inductive_branch):
    def energy(self,param_dict):
        return self.branch_flux**2/(2*param_dict[self.param_key])

    def __init__(self,start, end, param_key, branch_flux):
        inductive_branch.__init__(self,start, end, param_key, branch_flux)
        self.type_str = 'Inductor'

class jjunction(inductive_branch):
    def energy(self,param_dict):
        return param_dict[self.param_key]*(1-sp.cos(self.branch_flux))

    def __init__(self,start, end, param_key, branch_flux):
        inductive_branch.__init__(self,start, end, param_key, branch_flux)
        self.type_str = 'Josephson Junction'

class capacitor(capacitative_branch):
    def energy(self,param_dict):
        t = param_dict['t']
        return param_dict[self.param_key]*sp.diff(self.branch_flux, t)**2/2

    def __init__(self,start, end, param_key, branch_flux):
        capacitative_branch.__init__(self,start, end, param_key, branch_flux)
        self.type_str = 'Capacitor'

class circuit:
    def add_branch(self, start, end, type, param_key, param_val = None):
        switcher = {
            'Inductor' : inductor,
            'L' : inductor,
            'Capacitor' : capacitor,
            'C' : capacitor,
            'Josephson junction' : jjunction,
            'J' : jjunction,
            'JJ' : jjunction
        }
        assert type in switcher.keys(), 'Unknown branch type ' + type
        assert start in self.nodes+['ground'], 'Undefined node'
        assert end in self.nodes+['ground'], 'Undefined node'
        assert isinstance(param_key, str)
        t = self.param_dict['t']
        branch_type = switcher.get(type)
        branch_flux = 0
        if not end == 'ground':
            branch_flux += self.x[self.nodes.index(end)](t)
        if not start == 'ground':
            branch_flux -= self.x[self.nodes.index(start)](t)
        self.branches.append(branch_type(start, end, param_key, branch_flux))
        if not param_key in self.param_dict.keys():
            if param_val is None:
                self.param_dict[param_key] = sp.Symbol(param_key)
            else:
                self.param_dict[param_key] = param_val

    def remove_branch(self, branch):
        self.branches.remove(branch)

    def def_coord(self, coord, ind = None):
        if ind == None:
            ind = self.new_coord_index
        if not isinstance(coord[0], list):
            coord = [coord]
        for i in range(len(coord[0])):
            coord[0][i] = sp.nsimplify(coord[0][i])
        self.transformation[ind,:] = coord
        self.new_coord_index += 1

    def potential_symbolic(self, params = {}):
        for key in self.param_dict:
            if key not in params.keys():
                params[key] = self.param_dict[key]
        t = params['t']
        replacements=[(x(t), xnew) for x, xnew in zip(self.x,self.transformation.inv()*sp.Matrix(self.x))]
        return sum([b.energy(params) for b in self.branches if isinstance(b, inductive_branch)]).subs(replacements)

    def kinetic_symbolic(self, params = {}):
        for key in self.param_dict:
            if key not in params.keys():
                params[key] = self.param_dict[key]
        t = self.param_dict['t']
        replacements = [ ( sp.diff(x(t), t), v ) for x,v in zip(self.x, self.transformation.inv()*sp.Matrix(self.v))]
        return sum([b.energy(params) for b in self.branches if isinstance(b, capacitative_branch)]).subs(replacements)

    def legendre_transform(self, lagrangian):
        dLdv = [sp.diff(lagrangian,vi) for vi in self.v]
        dLdv = [self.C_eps*vi if dLdvi == 0 else dLdvi for dLdvi,vi in zip(dLdv,self.v)]
        eqs = [sp.simplify(pi - dLdvi) for pi,dLdvi in zip(self.p, dLdv) if not dLdvi == 0]
        if all([isinstance(b, capacitor) for b in self.branches if isinstance(b,capacitative_branch)]):
            sol_set = sp.linsolve(eqs, self.v)
        else:
            sol_set = sp.nonlinsolve(eqs, self.v)
        sol = [(vi, sol_i) for  vi,sol_i in zip(self.v,sol_set.args[0])]
        return (sum([pi*vi for pi,vi in zip(self.p,self.v)])-lagrangian).subs(sol)

    def SHO_hamiltonian(self, dims, taylor_order = 4, eliminate_coords = None):
        x = list(self.x)
        p = list(self.p)
        H = self.legendre_transform(self.kinetic_symbolic()-self.potential_symbolic())
        if not eliminate_coords == None:
            if not isinstance(eliminate_coords, list):
                eliminate_coords = [eliminate_coords]
            for ind in eliminate_coords:
                H = H.subs([(x.pop(ind), 0), (p.pop(ind), 0)])
        assert len(dims) == len(x)
        padded_dims = [d+int(np.floor(taylor_order/2)) for d in dims]
        P = 0
        for state in qt.state_number_enumerate(dims):
            P += qt.ket(state, dims)*qt.bra(state, padded_dims)
        vars = x+p
        T = taylor_sympy(H, vars, np.zeros(len(vars)), taylor_order)
        m_list = []
        k_list = []
        for ind,var in enumerate(vars):
            for coeff,k in T:
                if sum(k) == 2 and k[ind] == 2:
                    if var in x:
                        k_list.append(2*coeff)
                    if var in p:
                        m_list.append(1/(2*coeff))
        x_ops = []
        p_ops = []
        for ind,k,m,d in zip(range(len(k_list)),k_list, m_list, padded_dims):
            op = [qt.qeye(d) for d in padded_dims]
            op[ind] = 1j*(qt.create(d)-qt.destroy(d))/np.sqrt(2)
            p_ops.append([(k*m)**(1/4), qt.tensor(op)])
            op[ind] = (qt.create(d)+qt.destroy(d))/np.sqrt(2)
            x_ops.append([(k*m)**(-1/4), qt.tensor(op)])
        vars_ops = x_ops + p_ops
        coeffs = []
        ops = []
        param_keys = tuple(self.param_dict.keys())
        sym_params = [self.param_dict[key] for key in param_keys]
        for coeff, k in T:
            c = sp.lambdify(sym_params, coeff*sp.prod([op[0]**ki for op,ki in zip(vars_ops,k) if ki>0]))
            coeffs.append(lambda params, c = c: c(*params))
            ops.append(P*(np.prod([op[1]**ki for op,ki in zip(vars_ops, k) if ki>0]))*P.dag())
        return au.operator_generator(coeffs, ops, param_keys)

    def __str__(self):
        out = 'Nodes: ' + ', '.join(self.nodes)+'\n'
        for branch in self.branches:
            out += branch.__str__()+'\n'
        return out

    def __init__(self, nodes, transformed_nodes = [], transformation = []):
        assert isinstance(nodes,list)
        self.nodes = nodes
        self.branches = []
        self.param_dict = {}
        self.param_dict['t'] = sp.Symbol('t', real = True)
        self.x = sp.symbols('x0:'+str(len(nodes)))
        self.v = sp.symbols('v0:'+str(len(nodes)))
        self.p = sp.symbols('p0:'+str(len(nodes)))
        if transformation:
            self.transformation = sp.Matrix(transformation)
        else:
            self.transformation = sp.eye(len(nodes))
        self.new_coord_index = 0
        self.C_eps = sp.Symbol('C_eps')
