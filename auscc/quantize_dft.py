import qutip as qt
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from auscc.plottingtools import eval_wavefunction

def fourier_eval(F, x, xmin = None, xmax = None): # This is just a function a made to help me understand how the fourier transform can approximate the function that generated the sample
    if xmin == None:
        xmin = -np.pi
    if xmax == None:
        xmax = np.pi
    N = len(F)
    s = 2*np.pi*(1-1/N)/(xmax-xmin)
    theta = -2*np.pi*(1-1/N)*xmin/(xmax-xmin)
    for q in range(int(np.floor(N/2))):
        if q == 0:
            out = F[0]/N
        elif q == N/2:
            out += F[q]/(2*N)*(np.exp(1j*s*q*x)*np.exp(1j*q*theta)+np.exp(-1j*s*q*x)*np.exp(-1j*q*theta))
        else:
            out += F[q]/N*np.exp(1j*s*q*x)*np.exp(1j*q*theta)
            out += F[-q]/N*np.exp(-1j*s*q*x)*np.exp(-1j*q*theta)
    return out

def fourier_eval2(F, x, xmin = None, xmax = None): # This is just a function a made to help me understand how the fourier transform can approximate the function that generated the sample
    if xmin == None:
        xmin = -np.pi*np.ones(U.ndim)
    if xmax == None:
        xmax = np.pi*np.ones(U.ndim)
    N = F.shape
    s = [2*np.pi*(1-1/Nn)/(xmaxn-xminn) for Nn,xmaxn,xminn in zip(N,xmax,xmin)]
    theta = [-sn*xminn for sn,xminn in zip(s,xmin)]
    out = np.zeros(F.shape, dtype=complex)
    for k in np.ndindex(F.shape):
        outk = F[k]/np.prod(N)
        for kn, Nn, sn, thetan, xn in zip(k,N,s,theta, x):
            if kn > Nn/2:
                kn -= Nn
            if kn == Nn/2:
                outk *= (np.exp(1j*sn*kn*xn)*np.exp(1j*kn*thetan)+np.exp(-1j*sn*kn*xn)*np.exp(-1j*kn*thetan)) / 2
            else:
                outk *= np.exp(1j*sn*kn*xn)*np.exp(1j*kn*thetan)
        out += outk
    return out

def quantize_potential_dft(U, xmin = None, xmax = None):
    if xmin == None:
        xmin = -np.pi*np.ones(U.ndim)
    if xmax == None:
        xmax = np.pi*np.ones(U.ndim)
    UF = np.fft.fftn(U)
    N = U.shape
    s = [2*np.pi*(1-1/Nn)/(xmaxn-xminn) for Nn,xmaxn,xminn in zip(N,xmax,xmin)]
    theta = [-sn*xminn for sn,xminn in zip(s,xmin)]
    Nq = [int((Nn-1)/2) if Nn%2 else int(Nn/2) for Nn in N]
    E = []
    for Nn,Nqn,thetan in zip(N,Nq,theta):
        Ep = [np.exp(1j*thetan)*qt.qdiags(np.ones(2*Nqn), -1)]
        while len(Ep)<np.floor( (Nn - 1) / 2 ):
            Ep.append(Ep[-1]*Ep[0])
        Em = [Epn.dag() for Epn in Ep]
        Enyquist =  [] if (Nn%2) else [(Em[-1]*Em[1]+Ep[-1]*Ep[1])/2]
        E.append([qt.qeye(2*Nqn+1)]+Ep+Enyquist+list(reversed(Em)))
    out = 0
    for k in np.ndindex(N):
        ck = UF[k]/np.prod(N)
        ops = [En[kn] for kn,En in zip(k,E)]
        out += ck * qt.tensor(ops)
    return out, Nq

def quantize_1D_potential_dft(Un, xmin = None, xmax = None):
    if xmin == None:
        xmin = -np.pi
    if xmax == None:
        xmax = np.pi
    F = np.fft.fft(Un)
    N = len(Un)
    theta = -2*np.pi*(1-1/N)*xmin/(xmax-xmin)
    if (N % 2) == 0: # If N is even highest frequency is Nyquist, and we want both positive and negative so +1.
        Nq = int(N/2)
    else:
        Nq = int((N-1)/2)
    Ep = qt.qdiags(np.ones(2*Nq), -1)
    Eq = Ep
    for q in range(Nq):
        if q == 0:
            out = F[0]/N*qt.qeye(2*Nq+1)
        elif q == N/2:
            out += F[q]/(2*N)*(Eq*np.exp(1j*q*theta)+Eq.dag()*np.exp(-1j*q*theta))
        else:
            out += (F[q]*np.exp(1j*q*theta)/N)*Eq
            out += (F[-q]*np.exp(-1j*q*theta)/N)*Eq.dag()
            Eq = Eq*Ep
    return out, Nq

if __name__ == '__main__':
    N =128
    x = np.linspace(-np.pi,np.pi,N)
    dx = np.diff(x)[0]
    Uc = x**2
    Uq, Nq = quantize_potential_dft(Uc)
    Q = qt.charge(Nq[0])
    C = 5
    H = Q**2/(2*C)+Uq
    E,psi = H.eigenstates()
    p0 = abs(eval_wavefunction(psi[0], x))**2
    p1 = abs(eval_wavefunction(psi[1], x))**2
    plt.figure()
    plt.plot(x, Uc, color = 'k', linestyle = '-')
    plt.plot(x, E[0]*np.ones_like(x), color = 'k', linestyle = '--')
    plt.plot(x, (E[1]-E[0]) * p0 / (2*np.max(p0)) + E[0], color = 'b')
    plt.plot(x, E[1]*np.ones_like(x), color = 'k', linestyle = '--')
    plt.plot(x, (E[1]-E[0]) * p1 / (2*np.max(p1)) + E[1], color = 'r')


    # # Fourier fun
    N = 8
    x = sp.Symbol('x')
    y = sp.Symbol('y')
    f = sp.lambdify([x, y],x**2+y**2)
    xx,yy = np.meshgrid(np.linspace(-np.pi, np.pi, N),np.linspace(-np.pi, np.pi, N))
    dV = (2*np.pi/(N-1))**2
    U = f(xx,yy)
    UF, Nq = quantize_potential_dft(U)
    m = 1
    px = qt.tensor(qt.charge(Nq[0]), qt.qeye(2*Nq[1]+1))
    py = qt.tensor(qt.qeye(2*Nq[0]+1), qt.charge(Nq[1]))
    K = (px**2+py**2) / (2*m)
    E0, psi0 = (K+UF).groundstate()
    XX, YY = np.meshgrid(np.linspace(-np.pi, np.pi, 10*N),np.linspace(-np.pi, np.pi, 10*N))
    psix = eval_wavefunction(psi0, XX, YY)
    print(np.sum(np.abs(psix)**2)*dV)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.contourf(xx, yy, U, zdir='z', offset=-1, cmap=cm.coolwarm)
    ax.plot_wireframe(XX, YY, np.abs(psix)**2/np.max(np.abs(psix)**2))
    ax.set_xlabel('x')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylabel('y')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlabel('z')
    ax.set_zlim(-1, 1)
    plt.show()
