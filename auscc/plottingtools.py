import qutip as qt
import matplotlib.pyplot as plt
import numpy as np

def state_label(state):
    if state.isbra:
        state = state.dag()
    data = state.full()
    label = []
    while np.any(data,0):
        ind_max = np.argmax(np.absolute(data),0)
        label.append('({0:.2E})|{1}>'.format(data[ind_max[0],0], ''.join([str(k) for k in qt.state_index_number(state.dims[0],ind_max[0])])))
        data[ind_max[0],0] = 0
    return '+'.join(label)

class level:
    dx = 0.5
    color = 'k'
    picker_tol = 5
    def __str__(self):
        out = 'Level at E = {0:.3E}, {1}'.format(self.E, self.label)
        for t in self.transitions:
            out += '\n\t'+t.__str__()
        return out

    def plot(self, x_pos):
        self.x_pos = x_pos
        self.artist, = plt.plot([x_pos-self.dx, x_pos+self.dx], [self.E, self.E],color = self.color, picker = self.picker_tol)

    def __init__(self, E, ket, label = ''):
        self.E = E
        self.transitions = []
        self.ket = ket
        self.label = label

class transition:
    alpha_hidden = 0.1
    alpha_shown = 1.
    def hide(self):
        self.artist.set_alpha(self.alpha_hidden)

    def show(self):
        self.artist.set_alpha(self.alpha_shown)

    def plot(self):
        x = [lvl.x_pos for lvl in self.levels]
        y = [lvl.E for lvl in self.levels]
        self.artist, = plt.plot(x,y,alpha = self.alpha_shown)

    def __str__(self):
        out = 'J = {0:.3E}, Delta = {1:.3E}'.format(self.strength,np.abs(self.levels[1].E-self.levels[0].E))
        return out

    def __init__(self, strength, initial_lvl, final_lvl, one_way = False):
        if np.imag(strength) == 0:
            strength = np.real(strength)
        self.strength = strength
        self.levels = [initial_lvl, final_lvl]
        for lvl in self.levels:
            lvl.transitions.append(self)

class _subspace:
    def __init__(self, lvls):
        self.levels = lvls

class level_diagram:
    """Plots an interactive level diagram based on H.
    The energy levels are taken as the diagonal of H, and off-diagonal elemnts are taken to be interaction

    Parameters
    ----------
    H : type
        Description of parameter `H`.
    ops : type
        Description of parameter `ops`.

    """

    def plot(self):
        ax = self.ax
        plt.ylabel('Energy')
        for sub in self.deg_subspaces:
            spacing = 0.25
            L = sum([2*lvl.dx for lvl in sub.levels])+(len(sub.levels)-1)*spacing
            x = -L/2
            for lvl in sub.levels:
                lvl.plot(x+lvl.dx)
                x += 2*lvl.dx+spacing
        for t in self.transitions:
            t.plot()

    def on_pick(self,event):
        for lvl in self.levels:
            if lvl.artist == event.artist:
                picked_lvl = lvl
            else:
                for t in lvl.transitions:
                    t.hide()
        for t in picked_lvl.transitions:
            t.show()
        self.fig.canvas.draw()
        print(picked_lvl)


    def __init__(self, states, ops, state_labels = []):
        # -- Assertions --
        if isinstance(ops, qt.Qobj): # Make ops list if given as Qobj
            self.ops = [ops]
        else:
            self.ops = ops
        assert isinstance(self.ops,list)
        if state_labels:
            assert len(state_labels) == len(states)
        else:
            state_labels = ['psi_'+str(k) for k in range(len(states))]
        for state in states:
            assert state.isket

        # -- Initializing figure --
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.fig.canvas.callbacks.connect('pick_event', self.on_pick)
        # -- Building level instances --
        self.levels = []
        self.deg_subspaces = []
        self.transitions = []

        for psi,label in zip(states,state_labels):
            E = sum([qt.expect(oper,psi) for oper in self.ops])
            self.levels.append(level(E,psi,label))
        degeneracy_tol = (max([lvl.E for lvl in self.levels])-min([lvl.E for lvl in self.levels]))*0.5e-1
        lvls = self.levels.copy()
        while lvls:
            new_subspace = [lvl for lvl in lvls if abs(lvl.E-lvls[0].E)<degeneracy_tol]
            self.deg_subspaces.append(_subspace(new_subspace))
            for lvl in new_subspace:
                lvls.remove(lvl)
        # -- Building transition instances --
        min_strength = 1e-11
        for i in range(len(states)-1):
            for j in range(i+1,len(states)):
                strength = sum(oper.matrix_element(states[i],states[j]) for oper in self.ops)
                if np.abs(strength)>min_strength:
                    self.transitions.append(transition(strength, self.levels[i], self.levels[j]))




def expect_plot(result, ax = None, title = ''):
    if ax == None:
        fig = plt.figure()
        ax = plt.axes()
    plt.sca(ax)
    if isinstance(result.expect, dict):
        plt.rc('text', usetex=True)
        for label in result.expect:
            raw_label = r"$"+label+"$"
            plt.plot(result.times, result.expect[label], label = raw_label)
        plt.legend()
    else:
        for y in result.expect:
            plt.plot(result.times, y)
    plt.title(title)
    plt.xlabel(r"$ t $")
    plt.ylabel('Exp. value')
