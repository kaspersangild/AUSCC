import sympy as sp
def quantization_form(expr, initial_expand = True):
    """Brings a sympy expression on a form that the quantizers can understand.

    Parameters
    ----------
    expr : sympy expression
        The expression. Typically a symbolic potential or kinetic term
    Returns
    -------
    sympy expr
        The rewritten expression.

    """
    if initial_expand:
        expr = sp.expand(expr)
    if expr.func == sp.Add:
        expr = sp.Add(*[quantization_form(term,initial_expand = False) for term in expr.args])
    if expr.func == sp.Mul:
        expr = sp.Mul(*[quantization_form(factor,initial_expand = False) for factor in expr.args])
    if (expr.func == sp.cos) or (expr.func == sp.sin):
        if expr.args[0].func == sp.Add:
            arg_terms = expr.args[0].args
            arg_terms_dummy = sp.symbols('arg_dummy0:'+str(len(arg_terms)))
            # The reason for these subs shenanigans are to ensure that we don't mess with the coefficients of the arguments, so cos(2x)->cos(2x) and not cos(2x)->2*cos(x)**2-1
            expr = expr.subs([(arg, arg_dummy) for arg,arg_dummy in zip(arg_terms,arg_terms_dummy)])
            expr = sp.expand_trig(expr)
            expr = expr.subs([(arg_dummy, arg) for arg_dummy, arg in zip(arg_terms_dummy, arg_terms)])
    if initial_expand: # Used to cancel out terms
        expr = sp.expand(expr)
    return expr

def build_C_mat(branches, v, node_voltages): # This needs to be expanded upon once i do the control module properly
    Vg = list(set([b.bias_voltage for b in branches]))
    C_mat11 = sp.zeros(len(v))
    C_mat12 = sp.zeros(len(v), len(Vg))
    for b in branches:
        if b.type == 'Capacitor':
            b_voltage = node_voltages[b.end]-node_voltages[b.start]+b.bias_voltage
            c1 = sp.Matrix([b_voltage.coeff(vn) for vn in v]) # Vector with coefficients such that v1.T*v_vec = branch voltage
            c2 = sp.Matrix([b_voltage.coeff(Vgn) for Vgn in Vg])
            C_mat11 += b.symbol*c1*c1.T
            C_mat12 += b.symbol*c1*c2.T
    qg = C_mat12*sp.Matrix(Vg)
    return C_mat11, qg

def build_potential(branches, node_fluxes):
    node_fluxes = node_fluxes
    U = 0
    for b in branches:
        b_flux = node_fluxes[b.end]-node_fluxes[b.start]+b.bias_flux
        if b.type == 'Inductor':
            U += (b_flux)**2/(2*b.symbol)
        elif b.type == 'Josephson junction':
            U += b.symbol*(1-sp.cos(b_flux))
    return quantization_form(U)

def get_circuit_symbols(branches):
    circsyms = []
    for b in branches:
        for sym in b.symbol.free_symbols:
            if not sym in circsyms:
                circsyms.append(sym)
    return tuple(circsyms)

def get_control_symbols(branches):
    controlsyms = []
    for b in branches:
        if not b.bias_voltage == 0:
            for sym in b.bias_voltage.free_symbols:
                if not sym in controlsyms:
                    controlsyms.append(sym)
        if not b.bias_flux == 0:
            for sym in b.bias_flux.free_symbols:
                if not sym in controlsyms:
                    controlsyms.append(sym)
    return tuple(controlsyms)
