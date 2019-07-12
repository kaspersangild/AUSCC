from qutip import create, destroy, charge, qdiags, displace, qeye, tensor, num
from numpy import sqrt, ones, linspace, pi, conj, prod, flipud, concatenate
from numpy.fft import rfft
from sympy import lambdify, Symbol, Mul, Pow, cos, sin

def build_momentum_charge(dim, power):
    op = charge( ( dim-1 ) / 2 )**power
    return op

def build_position_charge(dim):
    x = Symbol('x_dummy')
    return build_pot_op_dft(dim, x)

def build_cos_charge(dim):
    op = (qdiags(ones(dim-1),1)+qdiags(ones(dim-1),-1))/2
    return op

def build_sin_charge(dim):
    op = 1j*(qdiags(ones(dim-1),1)-qdiags(ones(dim-1),-1))/2
    return op

# DFT method
def build_momentum_dft(dim, pow):
    return self.build_momentum_charge(dim)

def build_pot_op_dft(dim, sym_op):
    N = 2*dim-1 # Number of points where the potential is evaluated. The reasoning behind this is that the exponential operator exp(i*n*x) is nonzero in our subspace for 2*dim-1 different values of n. Thus to characterize all frequencies we need 2*dim-1 different points.
    x = linspace(0,2*pi*(N-1)/N , N)
    x[x>pi] = x[x>pi]-2*pi
    u = lambdify(sym_op.free_symbols, sym_op)(x)
    u_fft = rfft(u)
    E_n = qdiags(ones(dim-1),-1) # Operator representation of exp(i*n*x)
    E_1 = E_n # Operator representing exp(i*x)
    for n in range(len(u_fft)):
        if n == 0:
            op = u_fft[n]*qeye(dim)
        else:
            op += u_fft[n]*E_n
            op += conj(u_fft[n])*E_n.dag()
            E_n *= E_1
    return op/N

# SHO basis
def build_momentum_SHO(dim, power, zeta):
    if power == 1:
        op = 1j*sqrt(1/(2*zeta))*(create(dim)-destroy(dim))
    elif power == 2:
        op = -(create(dim)**2+destroy(dim)**2-2*num(dim)-1)/(2*zeta)
    return op

def build_position_SHO(dim, power, zeta):
    if power == 1:
        op = sqrt(zeta/2)*(create(dim)+destroy(dim))
    elif power == 2:
        op = zeta/2*(create(dim)**2+destroy(dim)**2+2*num(dim)+1)
    return op

def build_cos_SHO(dim, zeta, coeff):
    D = displace(dim,1j*float(coeff*sqrt(zeta/2))) # exp(coeff*1j*x)
    op = (D+D.dag())/2
    return op

def build_sin_SHO(dim, zeta, coeff):
    D = displace(dim,1j*float(coeff*sqrt(zeta/2)))
    op = (D-D.dag())/(2*1j)
    return op

def build_non_tensor_op(sym_op, x, p, dim, method, zeta, period):
    assert len(sym_op.free_symbols) == 1, 'Unexpected tensor operator:' + str(sym_op)
    # If momentum operator...
    if p in sym_op.free_symbols:
        # Only supports p and p**2
        if sym_op.func == Pow:
            power = sym_op.args[1]
            assert power == 2 and isinstance(sym_op.args[0], Symbol), 'Unsupported operator:'+str(sym_op)
        else:
            power = 1
            assert isinstance(sym_op, Symbol), 'Unsupported operator:'+str(sym_op)
        if method == 'dft':
            return build_momentum_charge(dim, power)
        elif method == 'sho':
            return build_momentum_SHO(dim, power, zeta)
    # If position operator...
    if x in sym_op.free_symbols:
        if method == 'dft':
            return build_pot_op_dft(dim, sym_op)
        elif method == 'sho':
            if sym_op.func == cos:
                coeff = sym_op.args[0]/x
                return build_cos_SHO(dim, zeta, coeff)
            elif sym_op.func == sin:
                coeff = sym_op.args[0]/x
                return build_sin_SHO(dim, zeta, coeff)
            elif sym_op.func == Pow:
                power = sym_op.args[1]
                return build_position_SHO(dim, power, zeta)
            elif sym_op.func == Symbol:
                power = 1
                return build_position_SHO(dim, power, zeta)
            else:
                raise ValueError('Unrecognized operator '+str(sym_op))




def build_operator(sym_op, x, p, dims, method, zeta, period):
    if sym_op.func ==  Mul:
        factors = sym_op.args
    else:
        factors = [sym_op]
    op_list = []
    for i in range(len(x)):
        if (x[i] in sym_op.free_symbols) or (p[i] in sym_op.free_symbols):
            new_op = 1
            for fac in factors:
                if (x[i] in fac.free_symbols) or (p[i] in fac.free_symbols):
                    new_op *= build_non_tensor_op(fac, x[i], p[i], dims[i], method[i], zeta[i], period[i])
        else:
            new_op = qeye(dims[i])
        op_list.append(new_op)
    return tensor(op_list)

class Operator_library:
    def __call__(self, sym_op):
        x = self.settings.coord_syms
        p = self.settings.momentum_syms
        dims = self.settings.dims
        method = self.settings.quantization_method
        zeta = self.settings.zeta
        period = self.settings.period
        for s in sym_op.free_symbols:
            assert (s in x) or (s in p), 'Unknown operator symbol '+str(s)
        sym_op = sym_op.evalf()
        if not sym_op in self.ops_dict.keys():
            self.ops_dict[sym_op] = build_operator(sym_op, x, p, dims, method, zeta, period)
        return self.ops_dict[sym_op]

    def __init__(self, settings):
        self.settings = settings
        self.ops_dict = dict()
