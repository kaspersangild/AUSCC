import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import itertools

def operator_decomposition(H):
    # Hamiltonian decomposition
    dims = H.dims[0]
    H_decomposition = []
    for N in range(len(dims)+1):
        # N = 0 -> constant term
        # N = 1 -> single mode terms
        # N = 2 -> 2 mode interaction
        # and so forth...
        for inds in itertools.combinations(iterable=range(len(dims)), r=N):
            if inds: # This if statement is just there because ptrace and tensor does not work if inds is empty
                h = H.ptrace(inds) / np.prod([d for i,d in enumerate(dims) if not i in inds])
                # Figuring out tensor re-ordering
                new_dims_order = [i for i in inds] + [j for j in range(len(dims)) if not j in inds] # This is the new order (inds[0], ind[1],..., notInds[0],...)
                reorder = [new_dims_order.index(i) for i in range(len(new_dims_order))] # This restores old order
                h_tensor = qt.tensor([h]+[qt.qeye(d) for i,d in enumerate(dims) if i not in inds]).permute(reorder) # Tensor product made in new_dims_order. Reordered by permute
            else:
                h = H.tr()/np.prod(dims)
                h_tensor = h*qt.qeye(list(dims))
            H -= h_tensor
            H_decomposition.append([inds,[h],[h_tensor]])
    if not H.tidyup()==qt.qzero(list(dims)):
        warnings.warn("Decomposition seems to be bad!")
    return H_decomposition

class qsystem:

    def simulate(self, rho0, tlist, e_ops=[], progress_bar = None):
        if callable(self.H):
            H = self.H(self.args)
        else:
            H = self.H
        return self.solver(H, rho0, tlist, c_ops = self.c_ops, e_ops=e_ops, args = self.args, options=self.solver_opts, progress_bar=progress_bar)

    def __init__(self, H, c_ops = [], args = {}, solver = qt.mesolve, solver_opts = qt.Options()):
        self.H = H
        self.c_ops = c_ops
        self.args = args
        self.solver = solver
        self.solver_opts = solver_opts
