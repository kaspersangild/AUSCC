from qutip import tensor, qeye, qzero
from numpy import sum, prod
from itertools import product, combinations

class Operator_decomposition:
    def single_mode_eigenstates(self,eigvals = 0):
        if eigvals == 0:
            eigvals = self.decomp[0][2].dims
        e_0 = self.decomp[0][1]
        E = []
        S = []
        for dec in self.decomp:
            if len(dec[0]) == 1:
                e,s = dec[1].eigenstates(eigvals = eigvals[dec[0][0]])
                E.append(e)
                S.append(s)
        labels = []
        full_states = []
        total_E = []
        for lab_ind in product(*[range(d) for d in eigvals]):
            full_states.append(tensor([s[n] for s,n in zip(S,lab_ind)]))
            total_E.append(sum(e[n]+e_0 for e,n in zip(E, lab_ind)))
            labels.append(lab_ind)
        return total_E, full_states, labels

    def get_from_inds(self, *inds):
        for dec_i in self.decomp:
            if len(dec_i[0]) == len(inds):
                if all([ind in dec_i[0] for ind in inds]):
                    return dec_i[1]

    def __str__(self):
        for dec_i in self.decomp:
            if len(dec_i[0]) == 2 or len(dec_i[0]) == 1:
                print(dec_i[0])
                print(dec_i[1])
                print('\n')
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
            for inds in combinations(iterable=range(len(dims)), r=N):
                if inds: # This if statement is just there because ptrace and tensor does not work if inds is empty
                    h = op.ptrace(inds) / prod([d for i,d in enumerate(dims) if not i in inds])
                    # Figuring out tensor re-ordering
                    new_dims_order = [i for i in inds] + [j for j in range(len(dims)) if not j in inds] # This is the new order (inds[0], ind[1],..., notInds[0],...)
                    reorder = [new_dims_order.index(i) for i in range(len(new_dims_order))] # This restores old order
                    h_tensor = tensor([h]+[qeye(d) for i,d in enumerate(dims) if i not in inds]).permute(reorder) # Tensor product made in new_dims_order. Reordered by permute
                else:
                    h = op.tr()/prod(dims)
                    h_tensor = h*qeye(list(dims))
                op -= h_tensor
                op_decomposition.append([inds,h,h_tensor])
        if not op.tidyup()==qzero(list(dims)):
            warnings.warn("Decomposition seems to be bad!")
        self.decomp = op_decomposition
