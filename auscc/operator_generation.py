import numpy as np
import qutip as qt
import sympy as sp
import dill as pickle
def _p_full(x_free, free_inds, X0):
    for x,i in zip(x_free, free_inds):
        X0[i] = x
    return X0

class opgen:
    """This is a class that can be used to express an qutip operator whith functions. It can be used if you have a set of operator with functions that can produce coefficients. Then by calling an instance of the class one can evaluate the operator for a given choice of parameters.
    Parameters
    ----------
    terms : list
        List of tuples on the form (coeff, op), where coeff is some callable object that produces the coefficient to the assosciated operater, op.
    """
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
        return sum(coeff(*args)*op for coeff,op in self.terms)
    def __init__(self, terms):
        self.terms = terms

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
        self.sym_terms = sym_terms
        symbols = set()
        symbols = symbols.union(*[coeff.free_symbols for coeff,_ in sym_terms])
        self.symbols = list(symbols)
        terms = [(sp.lambdify(symbols, sym_coeff), op) for sym_coeff,op in sym_terms]
        self.og = opgen(terms)


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
