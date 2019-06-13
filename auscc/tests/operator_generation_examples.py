import qutip as qt
import auscc as au
sx = [qt.tensor(qt.sigmax(),qt.qeye(2)), qt.tensor(qt.qeye(2),qt.sigmax())]
sy = [qt.tensor(qt.sigmay(),qt.qeye(2)), qt.tensor(qt.qeye(2),qt.sigmay())]
sz = [qt.tensor(qt.sigmaz(),qt.qeye(2)), qt.tensor(qt.qeye(2),qt.sigmaz())]
ops = [sz[0], sz[1], .5*(sx[0]*sx[1]+sy[0]*sy[1])]
coeffs = lambda *args, **kwargs: [1, args[0], kwargs['Jxx']]
td_strings = lambda *args, **kwargs: ['',kwargs['pulse'],'']
args = [1, 0.1]
kwargs = {'Jxx':0.01, 'pulse':'alpha*t', 'Ay':0.1}
og = au.OG(ops=ops, coeff_gen=coeffs, td_string_gen = td_strings)
og2 = au.OG([sx[0], sy[0]], lambda *args,**kwargs: [args[1], kwargs['Ay']], lambda *args,**kwargs: ['cos(w*t)','cos(w*t)'])
og_sum = og+og2
print(og(1,0.1, Jxx = 0.01, Ay = 0.1, pulse = 'alpha*t'))
print(og2(1,0.1, Jxx = 0.01, Ay = 0.1, pulse = 'alpha*t'))
print(og_sum(1,0.1, Jxx = 0.01, Ay = 0.1, pulse = 'alpha*t'))
