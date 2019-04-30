import auscc as au
import matplotlib.pyplot as plt
import sympy as sp

print('LC oscillator')
EJ,C,Cg,L = sp.symbols('E_J, C, C_g, L')
circ = au.circuit()
circ.add_branch(start=0, end=1, type='Capacitor', symbol = C)
circ.add_branch(start=0, end=1, type='Inductor', symbol = L)
print(circ)
H_og = circ.quantize()
print(H_og({L: 1, C:1}))


print('Simple grounded transmon')
circ = au.circuit()
circ.add_branch(start=0, end=1, type='Capacitor', symbol = C)
circ.add_branch(start=0, end=1, type='Josephson Junction', symbol = EJ)
print(circ)
H_og = circ.quantize()
print(H_og({EJ: 1, C:1}))


print('ZZ-coupler')
V = sp.Matrix([[1,-1,0,0],[0,0,1,-1],[1,1,-1,-1],[1,1,1,1]])/2
circ = au.circuit(V = V)
circ.add_branch(start=1, end=3, type='Capacitor', symbol=C)
circ.add_branch(start=1, end=3, type='JJ', symbol=EJ)
circ.add_branch(start=3, end=2, type='Capacitor', symbol=C)
circ.add_branch(start=3, end=2, type='JJ', symbol=EJ)
circ.add_branch(start=2, end=4, type='Capacitor', symbol=C)
circ.add_branch(start=2, end=4, type='JJ', symbol=EJ)
circ.add_branch(start=4, end=1, type='Capacitor', symbol=C)
circ.add_branch(start=4, end=1, type='JJ', symbol=EJ)
for i in range(1,5):
    circ.add_branch(start = 0, end = i, type = 'Capacitor', symbol = Cg)
# circ.def_coord([1/2,       -1/2,       0,          0])
# circ.def_coord([0,         0,          1/2,        -1/2])
# circ.def_coord([1/2,       1/2,        -1/2,       -1/2])
# circ.def_coord([1,         1,          1,          1])
print(circ)
H_og = circ.quantize(ignoreable_coordinates = [4])
# H_gen = circ.SHO_hamiltonian(dims=[2,2,2], taylor_order = 4, eliminate_coords = [3])
# print(au.operator_decomposition(H_gen({'t':0,EJ:5,C:1})),'\n\n')
