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
            'J' : jjunction
        }
        assert type in switcher.keys(), 'Unknown branch type ' + type
        assert start in self.nodes+['ground'], 'Undefined node'
        assert end in self.nodes+['ground'], 'Undefined node'
        assert isinstance(param_key, str)
        t = self.param_dict['t']
        branch_type = switcher.get(type)
        branch_flux = 0
        if not self.node_fluxes[end] == 0:
            branch_flux += self.node_fluxes[end](t)
        if not self.node_fluxes[start] == 0:
            branch_flux -= self.node_fluxes[start](t)
        self.branches.append(branch_type(start, end, param_key, branch_flux))
        if not param_key in self.param_dict.keys():
            if param_val is None:
                self.param_dict[param_key] = sp.Symbol(param_key)
            else:
                self.param_dict[param_key] = param_val

    def remove_branch(self, branch):
        self.branches.remove(branch)

    def potential_symbolic(self, params = {}):
        for key in self.param_dict:
            if key not in params.keys():
                params[key] = self.param_dict[key]
        t = params['t']
        replacements=[(self.node_fluxes[key](t),self.node_fluxes[key]) for key in self.nodes]
        return sum([b.energy(params) for b in self.branches if isinstance(b, inductive_branch)]).subs(replacements)

    def kinetic_symbolic(self, params = {}):
        for key in self.param_dict:
            if key not in params.keys():
                params[key] = self.param_dict[key]
        t = self.param_dict['t']
        replacements = [ ( sp.diff(self.node_fluxes[key](t), t), self.node_fluxes_der[key] ) for key in self.nodes]
        return sum([b.energy(params) for b in self.branches if isinstance(b, capacitative_branch)]).subs(replacements)

    def legendre_transform(self, lagrangian):
        v = [self.node_fluxes_der[key] for key in self.nodes]
        q = [sp.Symbol('q_'+key) for key in self.nodes]
        eqs = [qi - sp.diff(lagrangian,vi) for qi,vi in zip(q,v)]
        sol_set = sp.nonlinsolve(eqs, v)
        sol = [(vi, sol_i) for  vi,sol_i in zip(v,sol_set.args[0])]
        return (sum([qi*vi for qi,vi in zip(q,v)])-lagrangian).subs(sol), q

    def SHO_hamiltonian(self, dims, taylor_order = 4, pd=[]):
        x = [self.node_fluxes[key] for key in self.nodes]
        assert len(dims) == len(x)
        H, p = self.legendre_transform(self.kinetic_symbolic()-self.potential_symbolic())
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
        x_test = [pd[key] for key in param_keys]
        for coeff, k in T:
            c = sp.lambdify(sym_params, coeff*sp.prod([op[0]**ki for op,ki in zip(vars_ops,k) if ki>0]))
            coeffs.append(lambda params, c = c: c(*params))
            ops.append(P*(np.prod([op[1]**ki for op,ki in zip(vars_ops, k) if ki>0]))*P.dag())
        return au.operator_generator(coeffs, ops, param_keys)

    def __str__(self):
        out = 'Nodes: ' + ', '.join(self.node_keys)+'\n'
        for branch in self.branches:
            out += branch.__str__()+'\n'
        return out

    def __init__(self, nodes, branches = [], param_dict = {}):
        assert isinstance(nodes,list)
        self.nodes = nodes
        self.branches = branches
        self.param_dict = param_dict
        self.param_dict['t'] = sp.Symbol('t', real = True)
        self.node_fluxes = dict([(key, sp.Symbol('phi_'+key, real = True)) for key in nodes], real = True)
        self.node_fluxes_der = dict([(key, sp.Symbol('dot_phi_'+key, real = True)) for key in nodes], real = True)
        self.node_fluxes['ground'] = 0
