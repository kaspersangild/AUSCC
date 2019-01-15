import qutip as qt
import matplotlib.pyplot as plt
import numpy as np

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
