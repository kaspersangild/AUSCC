import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

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
