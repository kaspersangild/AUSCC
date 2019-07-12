from sympy import linsolve, FiniteSet, Symbol
from auscc.circuit.symbolics import get_control_symbols, get_circuit_symbols
from numpy import pi

def generate_p_syms(x_syms, momentum_symbol):
    p_syms = []
    for x in x_syms:
        x_string = str(x)
        if '_' in x_string:
            p_syms.append(Symbol(momentum_symbol + x_string[x_string.index('_'):]))
        else:
            p_syms.append(Symbol(momentum_symbol + '_' + x_string))
    return tuple(p_syms)

class Circuit_settings:
    """Class to keep track of the user defined settings of circuit, such as coordinates, symbol definitions, quantization settings etc.

    Parameters
    ----------
    Attributes
    ----------
    quantization_method : type
        Description of attribute `quantization_method`.
    dims : type
        Description of attribute `dims`.

    """

    def info_coord(self, latex = False):
        s = 'Flux nodes:\n'
        for fn in self.flux_node_syms:
            s += str(fn)+'\t'
        if self.coord_equations:
            s += '\n\nUsed coordinates:\n'
            for sym in self.coord_syms:
                s += str(sym)+'\t'
            s += '\n\nCoordinate definitions:\n'
            for eq in self.coord_equations:
                s+= str(eq[0])+' = '+str(eq[1])+'\n'
            s += '\nFlux nodes expressed using defined coordinates:\n'
            for fn in self.flux_node_syms:
                s += str(fn)+' = '+str(self.flux_node_expr[fn])+'\n'
        return s

    def define_coordinates_from_eqns(self, args):
        assert len(args) == len(self.flux_node_syms), 'Number of coordinates must equal number of nodes'
        equations = [arg[0]-arg[1] for arg in args]
        sol = linsolve(equations, *self.flux_node_syms)
        assert isinstance(sol, FiniteSet), 'Coordinates do not uniquely determine nodes'
        self.coord_equations = args
        self.coord_syms = tuple([arg[0] for arg in args if not arg[0] == 0])
        self.flux_node_expr = dict((fn, expr) for fn,expr in zip(self.flux_node_syms, sol.args[0]))
        self.momentum_syms = generate_p_syms(self.coord_syms, self.momentum_symbol)
        self.reset_quantization()

    def update(self, branches):
        if branches:
            self.circuit_symbols = get_circuit_symbols(branches)
            self.control_symbols = get_control_symbols(branches)

    def reset_quantization(self):
        self.quantization_method = len(self.coord_syms)*['dft']
        self.dims = len(self.coord_syms)*[15]
        self.zeta = [1]*len(self.coord_syms)
        self.period = [2*pi]*len(self.coord_syms)

    def __init__(self, flux_node_syms):
        # Symbolics + coordinates options
        self.momentum_symbol = 'q'
        self.flux_node_syms = flux_node_syms
        self.coord_syms = flux_node_syms
        self.coord_equations = []
        self.flux_node_expr = dict((fns , fns) for fns in flux_node_syms)
        self.momentum_syms = generate_p_syms(self.coord_syms, self.momentum_symbol)
        self.reset_quantization()
