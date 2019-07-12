def transform_state_unitary(state, U, inverse = False):
    if inverse:
        if state.isket:
            return U.dag()*state
        if state.isbra:
            return state*U
        if state.isoper:
            return U.dag()*state*U
    else:
        if state.isket:
            return U*state
        if state.isbra:
            return state*U.dag()
        if state.isoper:
            return U*state*U.dag()
