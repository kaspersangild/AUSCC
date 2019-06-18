import numpy as np
import qutip as qt
import sympy as sp
import dill as pickle
from collections.abc import Iterable
def _p_full(x_free, free_inds, X0):
    for x,i in zip(x_free, free_inds):
        X0[i] = x
    return X0

def isolate_td_factor(expr, t):
    td_factor = 1
    coeff = 1
    if expr.func == sp.Mul:
        for factor in expr.args:
            if t in factor.free_symbols:
                td_factor *= factor
            else:
                coeff *= factor
    return coeff,td_factor

def sympy2str(expr):
    safe_imag = sp.Symbol('__I__')
    expr_str = str(expr.subs(sp.I, safe_imag))
    expr_str = expr_str.replace('__I__', '1j')
    blacklisted_chararcters = ["{", "}", "\\", "_"]
    for b_char in blacklisted_chararcters:
        expr_str = expr_str.replace(b_char, '')
    return expr_str

def inv_mat_coeffs(mat):
    np.linalg.inv(mat)
    return k

class OG:
    """Base class for all operator generators. Everything else should essentially just be clever ways of building one of these. It contains method for calling and addition and its attributes are:
        ----------
        ops : list
            list of operators.
        coeff_gen : callable
            When called it must return a list of coefficients corresponding to ops
        td_string_gen : callable
            Optional. Must return a list of strings containing the time dependendent coefficient of each operator in ops.
    """
    def __add__(self, other):
        new_ops = self.ops.copy()+other.ops.copy()
        new_coeff_gen = lambda *args, **kwargs: self.coeff_gen(*args, **kwargs)+other.coeff_gen(*args, **kwargs)
        new_td_string_gen = lambda *args, **kwargs: self.td_string_gen(*args, **kwargs)+other.td_string_gen(*args, **kwargs)
        return OG(ops = new_ops, coeff_gen = new_coeff_gen, td_string_gen = new_td_string_gen)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __call__(self, *args, **kwargs):
        ops = self.ops_gen(*args, **kwargs)
        coeffs = self.coeff_gen(*args, **kwargs)
        td_strings = self.td_string_gen(*args, **kwargs)
        op_cnst = 0
        while any([s == '' for s in td_strings]):
            ind = np.argmax([s == '' for s in td_strings])
            op_cnst += coeffs.pop(ind)*ops.pop(ind)
            del td_strings[ind]
        if len(td_strings) == 0:
            return op_cnst
        else:
            if op_cnst == 0:
                return [[op,s] for s,op in zip(td_strings, ops)]
            else:
                return [op_cnst]+[[op,s] for s,op in zip(td_strings, ops)]

    def __init__(self, ops_gen, coeff_gen, td_string_gen = None):
        if td_string_gen == None:
            td_string_gen = lambda *args, **kwargs: len(ops)*['']
        self.ops_gen = ops_gen
        self.coeff_gen = coeff_gen
        self.td_string_gen = td_string_gen

class td_opgen:
    def __call__(self, *args):
        """Generates the operator with coefficients evaluated using args_dict. The generated operator is the sum of 'terms', with all coefficients evaluated with *args.

        Parameters
        ----------
        args : iterable
            Arguments to pass to the coefficient functions in 'terms'.

        Returns
        -------
        QObj
            Generated operator.

        """
        return [self.constant_og(*args)]+[[td_op(*args), td_str] for td_op, td_str in zip(self.td_og, self.td_strings)]
    def __init__(self, constant_terms, *td_terms):
        self.constant_og = opgen(constant_terms)
        self.td_strings = []
        self.td_og = []
        for td_T in td_terms:
            self.td_og.append(opgen(td_T[0]))
            self.td_strings.append(td_T[1])


class symopgen:
    """This is a class that can be used to express an qutip operator symbolically. It can be used if you have a set op operator with coefficients in the for of symbolic expressions. Then by calling an instance of the class one can evaluate the operator for a given choice of parameters.

    Parameters
    ----------
    sym_terms : list
        Each entry in on the form (sym_coeff,op) where sym_coeff is the coefficient as a sympy expression and op is the assosciated qutip qObj.

    """



    def __call__(self, args_dict):
        """Generates the operator with coefficients evaluated using args_dict. The generated operator is the sum of 'sym_terms', with all symbolic variables evaluated according to 'args_dict'.

        Parameters
        ----------
        args_dict : dictionary
            Dictionary with the symbols as keys and numeric values as values. It replaces the symbolic variable with the numeric value when evaluating the coefficients. If a symbols value is not not specified it is assumed to be zero.

        Returns
        -------
        QObj
            Generated operator.

        """
        args = np.zeros(len(self.symbols))
        for index, key in enumerate(self.symbols):
            if key in args_dict.keys():
                args[index] = args_dict[key]
        return self.og(*args)


    def __init__(self, sym_terms):
        if not isinstance(sym_terms[0], Iterable):
            self.sym_terms = [sym_terms]
        else:
            self.sym_terms = sym_terms
        symbols = set()
        symbols = symbols.union(*[coeff.free_symbols for coeff,_ in self.sym_terms])
        self.symbols = list(symbols)
        if len(self.sym_terms) == 0:
            self.og = lambda *x: 0
        else:
            terms = [(sp.lambdify(symbols, sym_coeff), op) for sym_coeff,op in self.sym_terms]
            self.og = opgen(terms)

class td_symopgen:
    def mesolve(self, args_dict, rho0, tlist, **kwargs):
        H = self(args_dict)
        mesolve_args = dict((sympy2str(symbol), val) for symbol, val in args_dict.items())
        return qt.mesolve(H, rho0, tlist, args = mesolve_args, **kwargs)
    def __call__(self, args_dict):
        """Generates the operator with coefficients evaluated using args_dict. The generated operator is the sum of 'sym_terms', with all symbolic variables evaluated according to 'args_dict'.

        Parameters
        ----------
        args_dict : dictionary
            Dictionary with the symbols as keys and numeric values as values. It replaces the symbolic variable with the numeric value when evaluating the coefficients. If a symbols value is not not specified it is assumed to be zero.

        Returns
        -------
        QObj
            Generated operator.

        """
        return [self.cnst_term(args_dict)]+[[og(args_dict), sympy2str(td_factor)] for og,td_factor in self.td_terms]


    def __init__(self, sym_terms, t_symbol):
        if not isinstance(sym_terms[0], Iterable):
            sym_terms = [sym_terms]
        else:
            sym_terms = sym_terms
        symbols = set()
        symbols = symbols.union(*[coeff.free_symbols for coeff,_ in sym_terms])
        self.symbols = list(symbols)
        td_factors = []
        cnst_term = []
        td_terms = []
        for T in sym_terms:
            coeff,td_factor = isolate_td_factor(T[0],t_symbol)
            if td_factor == 1:
                cnst_term.append(T)
            elif not td_factor in td_factors:
                td_factors.append(td_factor)
                td_terms.append([[coeff,T[1]], td_factor])
            else:
                for td_term in td_terms:
                    if td_term[1] == td_factor:
                        td_term[0].append([coeff, T[1]])
                        break
        self.cnst_term = symopgen(cnst_term)
        self.td_terms = [(symopgen(td_T[0]), td_T[1]) for td_T in td_terms]


class operator_generator:
    """This is a class that can generate a QuTiP operator for different parameters. An instance of the class can be called like instance(params), where params is a list or dictionary containing all the parameter values
    The operator returned when the class is called is:
    op = sum(coeffs[i](*params)*ops[i])

    Typically, one would build the coeffs functions using the sympy toolbox and lambdify.

    An obvious optimization trick would be to enable the class to create a cython version of the the coeffs function for a large speed-up.


    Parameters
    ----------
    coeffs : list
        List of callable coefficients. Should take params as input and output the coefficient for the corresponding operator.
    ops : type
        Description of parameter `ops`.
    """
    def save(self, filename):
        print('Saving to '+filename)
        file = open(filename, 'wb')
        pickle.dump(self, file, recurse=True)
        file.close()


    def fix_params(self,fixed_params = {}):
        fixed_inds = []
        x0 = [fixed_params[key] for key in fixed_params.keys()]
        X0 = np.zeros(len(self.keys))
        for key in fixed_params.keys():
            fixed_inds.append(self.keys.index(key))
        for ind,x in zip(fixed_inds, x0):
            X0[ind] = x
        free_inds = [ind for ind in range(len(self.keys)) if ind not in fixed_inds]
        keys_new = tuple([self.keys[ind] for ind in free_inds])
        coeffs_new = []
        for c in self.coeffs:
            coeffs_new.append(lambda x_free, X0 = X0, free_inds = free_inds, c = c : c(_p_full(x_free, free_inds, X0)))
        return operator_generator(coeffs_new, self.ops, keys_new)

    def __call__(self, params):
        if isinstance(params, dict):
            x = [params[key] for key in self.keys]
        else:
            x = params
        return np.sum([c(x)*op for c,op in zip(self.coeffs, self.ops)])

    def __init__(self, coeffs, ops, keys = ()):
        # -- Assertions --
        assert isinstance(coeffs, list) and isinstance(ops, list)
        self.coeffs = coeffs
        self.ops = ops
        self.keys = keys
