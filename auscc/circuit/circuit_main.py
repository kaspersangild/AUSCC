from auscc.circuit.symbolics import build_potential, build_C_mat
from auscc.circuit.settings import Circuit_settings
from auscc.circuit.evaluation import Circuit_evaluator
from auscc.circuit.operators import Operator_library
from sympy import Symbol, sympify, Matrix, MatrixSymbol
from numpy import ravel, ones, array, concatenate

class Branch:
    def __str__(self):
        out = '{0:20}{1:10} from {2} to {3}'.format(self.type, str(self.symbol), str(self.start), str(self.end))
        if not self.bias_flux == 0:
            out += ', Bias flux: '+str(self.bias_flux)
        if not self.bias_voltage == 0:
            out += ', Bias voltage: '+str(self.bias_voltage)
        return out

    def __init__(self, start, end, type, symbol, bias_voltage, bias_flux):
        self.start = start
        self.end = end
        self.type = type
        self.symbol = symbol
        self.bias_voltage = bias_voltage
        self.bias_flux = bias_flux

class Circuit:
    def add_branch(self, start, end, type, symbol,bias_voltage = 0, bias_flux = 0):
        assert (start in self.settings.flux_node_syms), 'Undefined node {}'.format(str(start))
        assert (end in self.settings.flux_node_syms), 'Undefined node {}'.format(str(end))
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
        self.settings.update(self.branches)

    def info_circ(self, latex = False):
        out = 'Circuit with '+str(len(self.settings.flux_node_syms))+' node(s)'
        out += '\n\nCircuit parameters:\n'
        for S in self.settings.circuit_symbols:
            out += str(S)+'\t'
        out += '\n\nControl parameters:\n'
        for S in self.settings.control_symbols:
            out += str(S)+'\t'
        out += '\n\nBranches:\n'
        for branch in self.branches:
            out += branch.__str__()+'\n'
        return out

    def info_quant(self, latex = False):
        out = 'Quantization Settings:\n'
        out += 'dims : '+str(self.settings.dims)
        out += '\nmethod : '+str(self.settings.quantization_method)
        out += '\nzeta : '+str(self.settings.zeta)
        out += '\nperiod : '+str(self.settings.period)
        out += '\n'
        return out

    def info(self, *args):
        out = ''
        for key in args:
            if key in ['Coordinates', 'coordinates', 'coord','coords', 'Coord', 'Coords', 'Coord']:
                out += self.settings.info_coord()
            elif key in ['Circuit', 'circuit', 'Circ', 'circ']:
                out += self.info_circ()
            elif key in ['Quantization', 'quantization', 'Quant', 'quant']:
                out += self.info_quant()
        print(out)

    def set_zeta(self, zeta):
        assert len(zeta) == len(self.settings.zeta), 'Wrong length of input list'
        self.settings.zeta = zeta
        self.ops_lib = Operator_library(self.settings)

    def set_dims(self, dims):
        assert len(dims) == len(self.settings.dims), 'Wrong length of input list'
        self.settings.dims = dims
        self.ops_lib = Operator_library(self.settings)

    def set_method(self, method):
        assert len(method) == len(self.settings.quantization_method), 'Wrong length of input list'
        self.settings.quantization_method = method
        self.ops_lib = Operator_library(self.settings)

    def define_coords(self,*args, method = 'eqs'):
        if method == 'eqs':
            self.settings.define_coordinates_from_eqns(*args)
        self.ops_lib = Operator_library(self.settings)

    def potential(self):
        return build_potential(self.branches, self.settings.flux_node_expr)

    def capacitance_matrix(self):
        return build_C_mat(self.branches, self.settings.coord_syms, self.settings.flux_node_expr)

    def build_evaluator(self):
        C_mat, qg = self.capacitance_matrix()
        self.evaluator = Circuit_evaluator( x_syms=self.settings.coord_syms,
                                            p_syms=self.settings.momentum_syms,
                                            U = self.potential(),
                                            C_mat=C_mat,
                                            qg=qg,
                                            circ_syms=self.settings.circuit_symbols,
                                            ctrl_syms=self.settings.control_symbols)

    def __call__(self, circ_subs, ctrl_subs = []):
        if self.evaluator == None:
            self.build_evaluator()
        return self.evaluator.eval_using_subs(ops_lib=self.ops_lib, circ_subs=circ_subs, ctrl_subs = ctrl_subs)

    def __init__(self, flux_node_symbols):
        self.settings = Circuit_settings(flux_node_symbols)
        self.ops_lib = Operator_library(self.settings)
        self.branches = []
        self.evaluator = None
