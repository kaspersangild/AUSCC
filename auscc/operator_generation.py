import numpy as np
import qutip as qt
import dill as pickle
def _p_full(x_free, free_inds, X0):
    for x,i in zip(x_free,free_inds):
        X0[i] = x
    return X0

class operator_generator:
    """This is a class that can generate a QuTiP operator for different parameters. An instance of the class can be called like instance(params), where params is a list containing all the parameter values
    The operator returned when the class is called is:
    op = sum(coeffs[i](params)*ops[i])

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
