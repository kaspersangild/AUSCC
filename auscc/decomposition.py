import qutip as qt
import numpy as np
import itertools

class operator_decomposition:
    def __str__(self):
        print(self.decomp)
        return ''

    def __init__(self, op):
        # Operator decomposition
        dims = op.dims[0]
        op_decomposition = []
        for N in range(len(dims)+1):
            # N = 0 -> constant term
            # N = 1 -> single mode terms
            # N = 2 -> 2 mode interaction
            # and so forth...
            for inds in itertools.combinations(iterable=range(len(dims)), r=N):
                if inds: # This if statement is just there because ptrace and tensor does not work if inds is empty
                    h = op.ptrace(inds) / np.prod([d for i,d in enumerate(dims) if not i in inds])
                    # Figuring out tensor re-ordering
                    new_dims_order = [i for i in inds] + [j for j in range(len(dims)) if not j in inds] # This is the new order (inds[0], ind[1],..., notInds[0],...)
                    reorder = [new_dims_order.index(i) for i in range(len(new_dims_order))] # This restores old order
                    h_tensor = qt.tensor([h]+[qt.qeye(d) for i,d in enumerate(dims) if i not in inds]).permute(reorder) # Tensor product made in new_dims_order. Reordered by permute
                else:
                    h = op.tr()/np.prod(dims)
                    h_tensor = h*qt.qeye(list(dims))
                op -= h_tensor
                op_decomposition.append([inds,h,h_tensor])
        if not op.tidyup()==qt.qzero(list(dims)):
            warnings.warn("Decomposition seems to be bad!")
        self.decomp = op_decomposition
