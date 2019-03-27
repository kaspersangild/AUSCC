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
    def add_branch(self, start, end, type, symbol):
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
                U += b.symbol*sp.cos(b_flux)
        K = (sp.Matrix(p).T*C_mat.inv()*sp.Matrix(p))[0,0]/2
        for cord in sorted(ignoreable_coordinates, reverse = True):
            K = K.subs(p[cord], 0)
            U = U.subs(x[cord], 0)
            del p[cord]
            del x[cord]
        return au.quantize(K, U, p, x, dims = dims, taylor_order = taylor_order, x0 = x0)





    # def remove_branch(self, branch):
    #     self.branches.remove(branch)

    # def def_coord(self, coord, ind = None):
    #     if ind == None:
    #         ind = self.new_coord_index
    #     if not isinstance(coord[0], list):
    #         coord = [coord]
    #     for i in range(len(coord[0])):
    #         coord[0][i] = sp.nsimplify(coord[0][i])
    #     self.transformation[ind,:] = coord
    #     self.new_coord_index += 1

    # def potential_symbolic(self, params = {}):
    #     for key in self.param_dict:
    #         if key not in params.keys():
    #             params[key] = self.param_dict[key]
    #     t = params['t']
    #     replacements=[(x(t), xnew) for x, xnew in zip(self.x,self.transformation.inv()*sp.Matrix(self.x))]
    #     return sum([b.energy(params) for b in self.branches if isinstance(b, inductive_branch)]).subs(replacements)

    # def kinetic_symbolic(self, params = {}):
    #     for key in self.param_dict:
    #         if key not in params.keys():
    #             params[key] = self.param_dict[key]
    #     t = self.param_dict['t']
    #     replacements = [ ( sp.diff(x(t), t), v ) for x,v in zip(self.x, self.transformation.inv()*sp.Matrix(self.v))]
    #     return sum([b.energy(params) for b in self.branches if isinstance(b, capacitative_branch)]).subs(replacements)

    # def legendre_transform(self, lagrangian):
    #     dLdv = [sp.diff(lagrangian,vi) for vi in self.v]
    #     dLdv = [self.param_dict['C_eps']*vi if dLdvi == 0 else dLdvi for dLdvi,vi in zip(dLdv,self.v)]
    #     eqs = [sp.expand(pi - dLdvi) for pi,dLdvi in zip(self.p, dLdv) if not dLdvi == 0]
    #     print(eqs)
    #     if all([isinstance(b, capacitor) for b in self.branches if isinstance(b,capacitative_branch)]):
    #         sol_set = sp.linsolve(eqs, self.v)
    #     else:
    #         sol_set = sp.nonlinsolve(eqs, self.v)
    #     print(sol_set)
    #     sol = [(vi, sol_i) for  vi,sol_i in zip(self.v,sol_set.args[0])]
    #     return (sum([pi*vi for pi,vi in zip(self.p,self.v)])-lagrangian).subs(sol)

    # def SHO_hamiltonian(self, dims, taylor_order = 4, eliminate_coords = None):
    #     x = list(self.x)
    #     p = list(self.p)
    #     H = self.legendre_transform(self.kinetic_symbolic()-self.potential_symbolic())
    #     if not eliminate_coords == None:
    #         if not isinstance(eliminate_coords, list):
    #             eliminate_coords = [eliminate_coords]
    #         for ind in eliminate_coords:
    #             H = H.subs([(x.pop(ind), 0), (p.pop(ind), 0)])
    #     assert len(dims) == len(x)
    #     padded_dims = [d+int(np.floor(taylor_order/2)) for d in dims]
    #     P = 0
    #     for state in qt.state_number_enumerate(dims):
    #         P += qt.ket(state, dims)*qt.bra(state, padded_dims)
    #     vars = x+p
    #     T = taylor_sympy(H, vars, np.zeros(len(vars)), taylor_order)
    #     m_list = []
    #     k_list = []
    #     for ind,var in enumerate(vars):
    #         for coeff,k in T:
    #             if sum(k) == 2 and k[ind] == 2:
    #                 if var in x:
    #                     k_list.append(2*coeff)
    #                 if var in p:
    #                     m_list.append(1/(2*coeff))
    #     x_ops = []
    #     p_ops = []
    #     for ind,k,m,d in zip(range(len(k_list)),k_list, m_list, padded_dims):
    #         op = [qt.qeye(d) for d in padded_dims]
    #         op[ind] = 1j*(qt.create(d)-qt.destroy(d))/np.sqrt(2)
    #         p_ops.append([(k*m)**(1/4), qt.tensor(op)])
    #         op[ind] = (qt.create(d)+qt.destroy(d))/np.sqrt(2)
    #         x_ops.append([(k*m)**(-1/4), qt.tensor(op)])
    #     vars_ops = x_ops + p_ops
    #     coeffs = []
    #     ops = []
    #     param_keys = tuple(self.param_dict.keys())
    #     sym_params = [self.param_dict[key] for key in param_keys]
    #     for coeff, k in T:
    #         c = sp.lambdify(sym_params, coeff*sp.prod([op[0]**ki for op,ki in zip(vars_ops,k) if ki>0]))
    #         coeffs.append(lambda params, c = c: c(*params))
    #         ops.append(P*(np.prod([op[1]**ki for op,ki in zip(vars_ops, k) if ki>0]))*P.dag())
    #     return au.operator_generator(coeffs, ops, param_keys)

    def __str__(self):
        N = max(max([b.start, b.end]) for b in self.branches)
        out = 'Circuit with '+str(N)+' node(s)\nBranches:\n'
        for branch in self.branches:
            out += branch.__str__()+'\n'
        return out

    def __init__(self, V = None):
        self.branches = []
        self.V = V
