import numpy as np
import sympy as sp
import math

def multinomial(n, k):
    out = math.factorial(n)
    for kt in k:
        out = out / math.factorial(kt)
    return out

def taylor_analytic(der_generator, x_0, N, tol = 1e-12):
    """
    Function that can generate multivariable taylor expansion of some function.
    Requires a function that can generate all derivatives of the function.

    Parameters
    ----------
    der_generator : function
        Function that takes x and k as inputs and returns the derivative  of the form
        (d^sum(k)/(dx1^k1 dx2^k2 ... dxm^km)f)(x_0). It should take inputs as:
        der_generator(x_0, k)
    x_0 : float
        Point around which the expansion is done
    N : integer
        The order to which the expansion should be done.

    Returns
    -------
    list of tuples on the form (c,k)
        Where c is the taylor coefficient for the term on the form:
            c * x_1^k[1] * x_2^[2] * ...
        Terms where c = 0 are discarded

    """
    # Building k_list
    def build_k_list(n,k = len(x_0)*[0], t = 0, k_list = []):
        if t == len(k)-1:
            k[t] = n
            k_list.append(k.copy())
        else:
            for kt in range(n+1):
                k[t] = kt
                build_k_list(n-kt,k, t+1)
        return k_list
    for n in range(N+1):
        k_list = build_k_list(n)

    # Building coefficients
    c_list = []
    for k in k_list:
        c_list.append(multinomial( n=sum(k),k=k) * der_generator(x_0,k) / math.factorial(sum(k)))
    return [(c,tuple(k)) for c,k in zip(c_list,k_list) if not ((isinstance(c, sp.Number) and c.is_zero) or (isinstance(c, float) and c < tol) )]


def taylor_sympy(f,x,x0,N, tol = 1e-12):
    """Short summary.

    Parameters
    ----------
    f : type
        Description of parameter `f`.
    x0 : type
        Description of parameter `x0`.
    N : type
        Description of parameter `N`.

    Returns
    -------
    type
        Description of returned object.

    """
    def der_generator(x0,k):
        der = f
        for xt,kt in zip(x,k):
            der = der.diff(xt,kt)
        der_lambda = sp.lambdify(x,der,"numpy")
        return der_lambda(*x0)
    return taylor_analytic(der_generator, x0, N, tol)



if __name__ == '__main__':
    def der_generator(x, k):
        # f(x1,x2) = 1+x1+x1*x2+x2^2
        x1 = x[0]
        x2 = x[1]
        if all([i == j for i,j in zip(k,[0,0])]):
            return 1.+x1+x1*x2+x2**2
        elif all([i == j for i,j in zip(k,[1,0])]):
            return 1. + x2
        elif all([i == j for i,j in zip(k,[0,1])]):
            return x1+2*x2
        elif all([i == j for i,j in zip(k,[1,1])]):
            return 1.
        elif all([i == j for i,j in zip(k,[0,2])]):
            return 2.
        else:
            return 0.
    res = taylor_analytic(der_generator, [0.,0.], 2)
    print(res)

    x = sp.symbols("x0 x1 ")
    f = 1+x[0]+x[0]*x[1]+x[1]**2
    T = taylor_sympy(f,x,[0.,0.],2)
    print(T)
