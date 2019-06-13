import auscc as au
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib import cm

N =128
x = np.linspace(-np.pi,np.pi,N)
dx = np.diff(x)[0]
Uc = 1-np.cos(x)
Uq, Nq = au.quantize_potential_dft(Uc)
Q = qt.charge(Nq[0])
C = 5
H = Q**2/(2*C)+Uq
E,psi = H.eigenstates()
p0 = abs(au.eval_wavefunction(psi[0], x))**2
p1 = abs(au.eval_wavefunction(psi[1], x))**2
p2 = abs(au.eval_wavefunction(psi[2], x))**2
plt.figure()
plt.plot(x, Uc, color = 'k', linestyle = '-')
plt.plot(x, E[0]*np.ones_like(x), color = 'k', linestyle = '--')
plt.plot(x, (E[1]-E[0]) * p0 / (2*np.max(p0)) + E[0], color = 'b')
plt.plot(x, E[1]*np.ones_like(x), color = 'k', linestyle = '--')
plt.plot(x, (E[2]-E[1]) * p1 / (2*np.max(p1)) + E[1], color = 'r')
plt.plot(x, E[2]*np.ones_like(x), color = 'k', linestyle = '--')
plt.plot(x, (E[3]-E[2]) * p2 / (2*np.max(p2)) + E[2], color = 'g')
plt.plot(x, (2*E[1]-E[0])*np.ones_like(x), color = 'k', linestyle = '--', alpha = 0.5)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$E/E_J$')

# # Fourier fun
N = 16
x = sp.Symbol('x')
y = sp.Symbol('y')
f = sp.lambdify([x, y],x**2+y**2)
xx,yy = np.meshgrid(np.linspace(-np.pi, np.pi, N/8),np.linspace(-np.pi, np.pi, 2*N))
dV = (2*np.pi/(N-1))**2
U = f(xx,yy)
UF, Nq = au.quantize_potential_dft(U)
m = 1
px = qt.tensor(qt.charge(Nq[0]), qt.qeye(2*Nq[1]+1))
py = qt.tensor(qt.qeye(2*Nq[0]+1), qt.charge(Nq[1]))
K = (px**2+py**2) / (2*m)
E0, psi0 = (K+UF).groundstate()
XX, YY = np.meshgrid(np.linspace(-np.pi, np.pi, 10*N),np.linspace(-np.pi, np.pi, 10*N))
psix = au.eval_wavefunction(psi0, XX, YY)
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


N = 16
x = sp.Symbol('x')
y = sp.Symbol('y')
f = sp.lambdify([x, y],-sp.cos(x)-sp.cos(y))
xx,yy = np.meshgrid(np.linspace(-np.pi, np.pi, N),np.linspace(-np.pi, np.pi, 2*N))
dV = (2*np.pi/(N-1))**2
U = f(xx,yy)
UF, Nq = au.quantize_potential_dft(U)
m = 0.25
px = qt.tensor(qt.charge(Nq[0]), qt.qeye(2*Nq[1]+1))
py = qt.tensor(qt.qeye(2*Nq[0]+1), qt.charge(Nq[1]))
K = (px**2+py**2) / (2*m)
E0, psi0 = (K+UF).groundstate()
XX, YY = np.meshgrid(np.linspace(-np.pi, np.pi, 10*N),np.linspace(-np.pi, np.pi, 10*N))
psix = au.eval_wavefunction(psi0, XX, YY)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contourf(XX, YY, f(XX,YY), zdir='z', offset=-1, cmap=cm.coolwarm)
ax.plot_wireframe(XX, YY, np.abs(psix)**2/np.max(np.abs(psix)**2))
ax.set_xlabel('x')
ax.set_xlim(-np.pi, np.pi)
ax.set_ylabel('y')
ax.set_ylim(-np.pi, np.pi)
ax.set_zlabel('z')
ax.set_zlim(-1, 1)
plt.show()
