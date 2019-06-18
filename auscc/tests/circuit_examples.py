import auscc as au
import matplotlib.pyplot as plt
import sympy as sp

print('LC oscillator')
EJ,C,Cg,L = sp.symbols('E_J, C, C_g, L')
circ = au.Circuit()
circ.add_branch(start=0, end=1, type='Capacitor', symbol = C)
circ.add_branch(start=0, end=1, type='Inductor', symbol = L)
print(circ.control_symbols())

#H_og = circ.quantize()
# circ.build_OG()
#print(H_og({L: 1, C:1}))


# print('Simple grounded transmon')
# circ = au.Circuit()
# circ.add_branch(start=0, end=1, type='Capacitor', symbol = C)
# circ.add_branch(start=0, end=1, type='Josephson Junction', symbol = EJ)
# #H_og = circ.quantize()
# #print(H_og({EJ: 1, C:1}))
#
#
# print('ZZ-coupler')
# circ = au.Circuit()
# circ.add_branch(start=1, end=3, type='Capacitor', symbol=C)
# circ.add_branch(start=1, end=3, type='JJ', symbol=EJ)
# circ.add_branch(start=3, end=2, type='Capacitor', symbol=C)
# circ.add_branch(start=3, end=2, type='JJ', symbol=EJ)
# circ.add_branch(start=2, end=4, type='Capacitor', symbol=C)
# circ.add_branch(start=2, end=4, type='JJ', symbol=EJ)
# circ.add_branch(start=4, end=1, type='Capacitor', symbol=C)
# circ.add_branch(start=4, end=1, type='JJ', symbol=EJ)
# print(circ)
# circ.V = sp.Matrix([[1,0,0,0,0],[0,1,-1,0,0],[0,0,0,1,-1],[0,1,1,-1,-1],[0,1,1,1,1]])/2
# circ.ignoreable_coordinates.append(4)
# #H_og = circ.quantize()
# #print(H_og({EJ: 1, C:1}))
