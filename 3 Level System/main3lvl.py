from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check, Qobj
import matplotlib.pyplot as plt
from Atoms3lvl import *
from Driving3lvl import *
import numpy as np
from scipy import integrate

N = 1

omega = 2. * np.pi * 0

Omega_R = 2 * np.pi * 1

J = 2 * np.pi * 5

bandwidth = 20

sampling_rate = 1000
endtime = 1
timesteps = int(endtime * sampling_rate)
t1timesteps = 2
t2timesteps = 200

gamma1 = 0

pertubation_length = endtime / 1

t1 = np.linspace(0, endtime, t1timesteps)
t2 = np.linspace(0, 3, t2timesteps)

noise_amplitude = 1.000

perturb_times = np.linspace(0, pertubation_length, t1timesteps)
random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

#S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(noise_amplitude, perturb_times, omega, bandwidth))

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, 0, N), sigmaz(0, N - 1, N), upup(0, N),
        sigmap(0, 0, N), sigmam(0, 0, N), downdown(0, N), anan(0, N)]

Commutatorlist = []
Anticommutatorlist = []

opts = Options(store_states=True, store_final_state=True, nsteps=10**9)

diag = simdiag([H0(omega, Omega_R, J, N)])

print(diag)

print(productstateX(0, N - 1, N))

Temperature = 1

Z = np.sum(np.exp(-diag[0][0]/(2*10**4*Temperature)))   #Omegas in MHz, T in K

#print(diag[0])

#print(diag[0][0])

#print(diag[1][0])

print(Z)

density_matrix = np.exp(-diag[0][0][0]/(2*10**4*Temperature))/Z * diag[1][0]*diag[1][0].dag()

#print(density_matrix)

for n in range(1, len(diag[1:])):
    density_matrix += np.exp(-diag[0][0][n]/(2*10**4*Temperature))/Z * diag[1][n]*diag[1][n].dag()



#print(density_matrix)

state1=Qobj([[0],
 [0],
 [0],
 [0],
 [0],
 [7.0711e-01],
 [0],
 [-7.0711e-01],
 [0]])

state1=Qobj(
[[0],
 [0],
 [0],
 [0],
 [-1.337e-01],
 [6.943e-01],
 [0],
 [6.9434e-01],
 [-1.337e-01]])

Temperature=10**(-5)

state = np.exp(-diag[0][0][0]/(2*10**4*Temperature))/Z  * diag[1][0]

#print(density_matrix)

for n in range(1, len(diag[1:])):
    state += np.exp(-diag[0][0][n]/(2*10**4*Temperature))/Z * diag[1][n]


#s2 = Qobj(state.data.toarray().reshape((9,1)),
#            dims=[[3,3],[1,1]])

#print(s2)

#print(productstateX(0, N - 1, N))

#print(state)

result_t1 = mesolve(H0(omega, Omega_R, J, N), productstateX(0, N - 1, N), t1, [], Exps, options=opts)
#result_t1 = mesolve(H0(omega, Omega_R, J, N), s2, t1, [], Exps, options=opts)




#print(spin_coherent(N, 2, 2, type='ket'))

#result_t1 = mesolve(H0(omega, Omega_R, J, N), thermal_dm(N,N), t1, [], Exps, options=opts)

result_t1t2 = mesolve(H0(omega, Omega_R, J, N), result_t1.states[- 1], t2, [], Exps, options=opts)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

#plt.rcParams.update({
#  "text.usetex": True,
#})


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('font',**{'family':'serif','serif':['Latin Modern Roman']})

#plt.rc('figure', figsize=(11.69/1.7, 8.27))

ax[0,0].errorbar(t1, np.real(result_t1.expect[0]), label="MagX")
ax[0,0].errorbar(t1, np.real(result_t1.expect[1]), label="MagZ")
ax[0,0].errorbar(t1, np.real(result_t1.expect[2]), label="MagY")
#ax[0].plot(t2, np.imag(Commutatorlist), label="Im(Commutator)")

#ax[1].plot(t2, np.real(Anticommutatorlist), label="Re(Anticommutator)")
#ax[1].plot(t2, np.imag(Anticommutatorlist), label="Im(Anticommutator)")

ax[0,0].legend()
#ax[1].legend()
ax[0,0].set_xlabel('t_1')
#ax[1].set_xlabel('t_measure - t_perturb')
plt.show()

Perturb = MagnetizationZ(N)
Measure = MagnetizationZ(N)


#print("Pertubed=", Perturb*result_t1.states[timesteps - 1])

state=Qobj([[0.00000000e+00],
 [-2.e-18],
 [2.e-17],
 [3.e-16],
 [1.e-33],
 [7.e-31],
 [2.e-16],
 [-7.e-31],
 [-2.e-35]])


s2 = Qobj(state.data.toarray().reshape((9, 1)),
            dims=[[3, 3], [1, 1]])


result_AB = mesolve(H0(omega, Omega_R, J, N), Perturb * result_t1.states[- 1], t2, [], Exps, options=opts)

dm = 0

#Commutator= (Measure* result_t1t2.states[t - 1] ).tr()

for t in range(0, t2timesteps):
    if dm == 1:
        prod_AB = result_t1t2.states[t - 1].tr() * (Measure * result_AB.states[t - 1]).tr()

        prod_BA = result_AB.states[t - 1].tr() * (Measure * result_t1t2.states[t - 1]).tr()


        Commutator = prod_AB - prod_BA

        AntiCommutator = prod_AB + prod_BA

        Commutatorlist.append(Commutator)
        Anticommutatorlist.append(AntiCommutator)

    else:
        prod_AB = result_t1t2.states[t - 1].dag() * Measure * result_AB.states[t - 1]

        prod_BA = result_AB.states[t - 1].dag() * Measure * result_t1t2.states[t - 1]

        Commutator = prod_AB - prod_BA

        print(Commutator)

        AntiCommutator = prod_AB + prod_BA

        Commutatorlist.append(-1.j*Commutator[0][0][0])
        Anticommutatorlist.append(AntiCommutator[0][0][0])
        # print('Commutator:', 1j * Commutator[0][0])
        # print('AntiCommutator: ', AntiCommutator[0][0])


fig, ax = plt.subplots(2, 2)

#ax[1].errorbar(t2[1:len(t2)], np.real(Commutatorlist[1:len(t2)]), label="Re(Commutator)", color="black")
#ax[1].plot(t2[1:len(t2)], np.imag(Anticommutatorlist[1:len(t2)]), label="Im(Anticommutator)")

ax[0,0].plot(t2[1:len(t2)], Commutatorlist[1:len(t2)], label=r"Commutator $ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$", color="black")
ax[0,0].plot(t2[1:len(t2)], Anticommutatorlist[1:len(t2)], label=r"Anti-commutator $ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$", color="#85bb65")

ax[0,0].set_xlabel(r'Time [$1/\Omega$]', fontsize=18)
#ax[1].set_xlabel('t_measure - t_perturb')

ax[0,0].set_ylabel(r'Expectation Value', fontsize=18)
ax[0,0].legend( fontsize=16) #loc="lower right",
#ax[1, 1].set_xlim([-1.5, 1.5])
ax[0,0].tick_params(axis="both", labelsize=16)


#plt.show()



omegas = np.linspace(-1.5*Omega_R, 1.5*Omega_R, num=1500)

integrals = []
integrals0 = []

x1 = np.linspace(t2[1], t2[-1])#np.array(x0[0:9])

y0 = np.array(Anticommutatorlist)

y31 = np.array(Commutatorlist)

for o in omegas:
    integrals.append(2*np.pi*integrate.simps(y31*np.exp(-1j*o*t2*2*np.pi), t2))

for o in omegas:
    integrals0.append(2*np.pi*integrate.simps(y0*np.exp(-1j*o*t2*2*np.pi), t2))

Temp=25*10**(-6)

#freq = np.fft.fftfreq(t2[1:len(t2)].shape[-1])*Omega_R
ax[1,0].plot(omegas, np.imag(integrals), linestyle='-', marker='o', markersize='0', label=r"$ Im(FT(\langle [ \sigma_z(0),\sigma_z(t) ] \rangle))$", color="black")
#ax[1].plot(omegas, np.imag(integrals), linestyle='-', marker='o', markersize='0', label=r"$ FT(\langle [ \sigma_z(0),\sigma_z(t) ] \rangle)$", color="grey")
ax[1,0].plot(omegas, np.real(integrals0), linestyle='-', marker='o', markersize='0', label=r"$ Re(FT(\langle \{ \sigma_z(0),\sigma_z(t) \} \rangle))$", color="#85bb65")

#ax[1].plot(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*integrals0, linestyle='-', marker='o', markersize='0', label=r"$ FT(\langle \{ \sigma_z(0),\sigma_z(t) \} \rangle)$", color="purple")


ax[1,0].set_xlabel(r'Frequency $\omega$ [$\Omega$]', fontsize=18)
#ax[1].set_xlabel('t_measure - t_perturb')

#ax[0].set_ylabel(r'Expectation Value', fontsize=18)
ax[1,0].legend( fontsize=16) #loc="lower right",
#ax[1].set_xlim([-2.5, 2.5])
ax[1,0].tick_params(axis="both", labelsize=16)
ax[1,0].set_ylabel(r'Correlation Spectrum', fontsize=18)
#plt.xlim(-2 / Omega_R, 2 / Omega_R)
#plt.legend()
#plt.show()


N = 2

omega = 2. * np.pi * 0

Omega_R = 2 * np.pi * 1

J = 2 * np.pi * 5

bandwidth = 20

sampling_rate = 1000
endtime = 1
timesteps = int(endtime * sampling_rate)
t1timesteps = 2
t2timesteps = 200

gamma1 = 0

pertubation_length = endtime / 1

t1 = np.linspace(0, endtime, t1timesteps)
t2 = np.linspace(0, 3, t2timesteps)

noise_amplitude = 1.000

perturb_times = np.linspace(0, pertubation_length, t1timesteps)
random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

#S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(noise_amplitude, perturb_times, omega, bandwidth))

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, 0, N), sigmaz(0, N - 1, N), upup(0, N),
        sigmap(0, 0, N), sigmam(0, 0, N), downdown(0, N), anan(0, N)]

Commutatorlist = []
Anticommutatorlist = []

opts = Options(store_states=True, store_final_state=True, nsteps=10**9)

diag = simdiag([H0(omega, Omega_R, J, N)])

print(diag)

print(productstateX(0, N - 1, N))

Temperature = 1

Z = np.sum(np.exp(-diag[0][0]/(2*10**4*Temperature)))   #Omegas in MHz, T in K

#print(diag[0])

#print(diag[0][0])

#print(diag[1][0])

print(Z)

density_matrix = np.exp(-diag[0][0][0]/(2*10**4*Temperature))/Z * diag[1][0]*diag[1][0].dag()

#print(density_matrix)

for n in range(1, len(diag[1:])):
    density_matrix += np.exp(-diag[0][0][n]/(2*10**4*Temperature))/Z * diag[1][n]*diag[1][n].dag()



#print(density_matrix)

state1=Qobj([[0],
 [0],
 [0],
 [0],
 [0],
 [7.0711e-01],
 [0],
 [-7.0711e-01],
 [0]])

state1=Qobj(
[[0],
 [0],
 [0],
 [0],
 [-1.337e-01],
 [6.943e-01],
 [0],
 [6.9434e-01],
 [-1.337e-01]])

Temperature=10**(-5)

state = np.exp(-diag[0][0][0]/(2*10**4*Temperature))/Z  * diag[1][0]

#print(density_matrix)

for n in range(1, len(diag[1:])):
    state += np.exp(-diag[0][0][n]/(2*10**4*Temperature))/Z * diag[1][n]


#s2 = Qobj(state.data.toarray().reshape((9,1)),
#            dims=[[3,3],[1,1]])

#print(s2)

#print(productstateX(0, N - 1, N))

#print(state)

result_t1 = mesolve(H0(omega, Omega_R, J, N), productstateX(0, N - 1, N), t1, [], Exps, options=opts)
#result_t1 = mesolve(H0(omega, Omega_R, J, N), s2, t1, [], Exps, options=opts)




#print(spin_coherent(N, 2, 2, type='ket'))

#result_t1 = mesolve(H0(omega, Omega_R, J, N), thermal_dm(N,N), t1, [], Exps, options=opts)

result_t1t2 = mesolve(H0(omega, Omega_R, J, N), result_t1.states[- 1], t2, [], Exps, options=opts)


plt.rcParams.update({
  "text.usetex": True,
})


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Latin Modern Roman']})

plt.rc('figure', figsize=(11.69/1.7, 8.27))


Perturb = MagnetizationZ(N)
Measure = MagnetizationZ(N)


#print("Pertubed=", Perturb*result_t1.states[timesteps - 1])

state=Qobj([[0.00000000e+00],
 [-2.e-18],
 [2.e-17],
 [3.e-16],
 [1.e-33],
 [7.e-31],
 [2.e-16],
 [-7.e-31],
 [-2.e-35]])


s2 = Qobj(state.data.toarray().reshape((9, 1)),
            dims=[[3, 3], [1, 1]])


result_AB = mesolve(H0(omega, Omega_R, J, N), Perturb * result_t1.states[- 1], t2, [], Exps, options=opts)

dm = 0

#Commutator= (Measure* result_t1t2.states[t - 1] ).tr()

for t in range(0, t2timesteps):
    if dm == 1:
        prod_AB = result_t1t2.states[t - 1].tr() * (Measure * result_AB.states[t - 1]).tr()

        prod_BA = result_AB.states[t - 1].tr() * (Measure * result_t1t2.states[t - 1]).tr()


        Commutator = prod_AB - prod_BA

        AntiCommutator = prod_AB + prod_BA

        Commutatorlist.append(Commutator)
        Anticommutatorlist.append(AntiCommutator)

    else:
        prod_AB = result_t1t2.states[t - 1].dag() * Measure * result_AB.states[t - 1]

        prod_BA = result_AB.states[t - 1].dag() * Measure * result_t1t2.states[t - 1]

        Commutator = prod_AB - prod_BA

        print(Commutator)

        AntiCommutator = prod_AB + prod_BA

        Commutatorlist.append(-1.j*Commutator[0][0][0])
        Anticommutatorlist.append(AntiCommutator[0][0][0])
        # print('Commutator:', 1j * Commutator[0][0])
        # print('AntiCommutator: ', AntiCommutator[0][0])



#ax[1].errorbar(t2[1:len(t2)], np.real(Commutatorlist[1:len(t2)]), label="Re(Commutator)", color="black")
#ax[1].plot(t2[1:len(t2)], np.imag(Anticommutatorlist[1:len(t2)]), label="Im(Anticommutator)")

ax[0,1].plot(t2[1:len(t2)], Commutatorlist[1:len(t2)], label=r"Commutator $ \langle [ \sigma_z(0),\sigma_z(t) ] \rangle$", color="black")
ax[0,1].plot(t2[1:len(t2)], Anticommutatorlist[1:len(t2)], label=r"Anti-commutator $ \langle \{ \sigma_z(0),\sigma_z(t) \} \rangle$", color="#85bb65")

ax[0,1].set_xlabel(r'Time [$1/\Omega$]', fontsize=18)
#ax[1].set_xlabel('t_measure - t_perturb')

ax[0,1].set_ylabel(r'Expectation Value', fontsize=18)
ax[0,1].legend( fontsize=16) #loc="lower right",
#ax[1, 1].set_xlim([-1.5, 1.5])
ax[0,1].tick_params(axis="both", labelsize=16)


#plt.show()



omegas = np.linspace(-1.5*Omega_R, 1.5*Omega_R, num=1500)

integrals = []
integrals0 = []

x1 = np.linspace(t2[1], t2[-1])#np.array(x0[0:9])

y0 = np.array(Anticommutatorlist)

y31 = np.array(Commutatorlist)

for o in omegas:
    integrals.append(2*np.pi*integrate.simps(y31*np.exp(-1j*o*t2*2*np.pi), t2))

for o in omegas:
    integrals0.append(2*np.pi*integrate.simps(y0*np.exp(-1j*o*t2*2*np.pi), t2))

Temp=25*10**(-6)

#freq = np.fft.fftfreq(t2[1:len(t2)].shape[-1])*Omega_R
ax[1,1].plot(omegas, np.imag(integrals), linestyle='-', marker='o', markersize='0', label=r"$ Im(FT(\langle [ \sigma_z(0),\sigma_z(t) ] \rangle))$", color="black")
#ax[1].plot(omegas, np.imag(integrals), linestyle='-', marker='o', markersize='0', label=r"$ FT(\langle [ \sigma_z(0),\sigma_z(t) ] \rangle)$", color="grey")
ax[1,1].plot(omegas, np.real(integrals0), linestyle='-', marker='o', markersize='0', label=r"$ Re(FT(\langle \{ \sigma_z(0),\sigma_z(t) \} \rangle))$", color="#85bb65")

#ax[1].plot(omegas, (1 - 2/(np.exp(2*omegas/Temp/10**4/6.558) + 1))*integrals0, linestyle='-', marker='o', markersize='0', label=r"$ FT(\langle \{ \sigma_z(0),\sigma_z(t) \} \rangle)$", color="purple")


ax[1,1].set_xlabel(r'Frequency $\omega$ [$\Omega$]', fontsize=18)
#ax[1].set_xlabel('t_measure - t_perturb')

#ax[0].set_ylabel(r'Expectation Value', fontsize=18)
ax[1,1].legend( fontsize=16) #loc="lower right",
#ax[1].set_xlim([-2.5, 2.5])
ax[1,1].tick_params(axis="both", labelsize=16)
ax[1,1].set_ylabel(r'Correlation Spectrum', fontsize=18)
#plt.xlim(-2 / Omega_R, 2 / Omega_R)
#plt.legend()
plt.show()

'''
freq = np.fft.fftfreq(t2[1:len(t2)].shape[-1])*Omega_R
plt.plot(freq, np.fft.fft(Commutatorlist[1:len(t2)]), linestyle='--', marker='o', markersize='5', label="Commutator")
plt.plot(freq, np.fft.fft(Anticommutatorlist[1:len(t2)]), linestyle='--', marker='o', markersize='5', label="Anticommutator")
#plt.xlim(-2 / Omega_R, 2 / Omega_R)
plt.legend()
plt.show()
'''


spectra_cb = [ohmic_spectrum]
a_ops = []

S = Cubic_Spline(perturb_times[0], perturb_times[-1], func(perturb_times, omega))

result_br = brmesolve(H0(omega, J, N), result_t1.states[timesteps - 1], perturb_times, a_ops, spectra_cb=spectra_cb,
                      options=opts)

result_t1t2_br = mesolve(H0(omega, J, N), result_br.states[timesteps - 1], t2, [], Exps, options=opts)

result_me = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], result_t1.states[timesteps - 1],
                    perturb_times, [bandwidth * sigmap(1, 0, N) / 10, bandwidth * sigmam(1, 0, N) / 10], Exps,
                    options=opts)

result_t1t2_me = mesolve(H0(omega, J, N), result_me.states[timesteps - 1], t2, [], Exps, options=opts)

anan = np.zeros(timesteps)
upup = np.zeros(timesteps)
downdown = np.zeros(timesteps)
updown = np.zeros(timesteps)
downup = np.zeros(timesteps)
anant1t2me = np.zeros(timesteps)
upupt1t2me = np.zeros(timesteps)
downdownt1t2me = np.zeros(timesteps)
updownt1t2me = np.zeros(timesteps)
downupt1t2me = np.zeros(timesteps)
upupbr = np.zeros(timesteps)
downdownbr = np.zeros(timesteps)
updownbr = np.zeros(timesteps)
downupbr = np.zeros(timesteps)
upupt1t2br = np.zeros(timesteps)
downdownt1t2br = np.zeros(timesteps)
updownt1t2br = np.zeros(timesteps)
downupt1t2br = np.zeros(timesteps)

if N == 1:
    for t in range(0, timesteps):
        anan[t] = np.real(result_me.states[t][0][0][0])
        upup[t] = np.real(result_me.states[t][1][0][1])
        downdown[t] = np.real(result_me.states[t][2][0][2])
        updown[t] = np.real(result_me.states[t][1][0][1])
        downup[t] = np.real(result_me.states[t][2][0][1])
        anant1t2me[t] = np.real(result_t1t2_me.states[t][0][0][0])
        upupt1t2me[t] = np.real(result_t1t2_me.states[t][1][0][1])
        downdownt1t2me[t] = np.real(result_t1t2_me.states[t][2][0][2])
        updownt1t2me[t] = np.real(result_t1t2_me.states[t][1][0][2])
        downupt1t2me[t] = np.real(result_t1t2_me.states[t][2][0][1])
        upupbr[t] = np.real(result_br.states[t][0][0][0])
        downdownbr[t] = np.real(result_br.states[t][1][0][1])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][0])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t][0][0][1])
        upupt1t2br[t] = np.real(result_t1t2_br.states[t][0][0][0])
        downdownt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][1])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][0])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t][0][0][1])
else:
    for t in range(0, timesteps):
        anan[t] = np.real(result_me.states[t].ptrace(0)[0][0][0])
        upup[t] = np.real(result_me.states[t].ptrace(0)[1][0][1])
        downdown[t] = np.real(result_me.states[t].ptrace(0)[2][0][2])
        updown[t] = np.real(result_me.states[t].ptrace(0)[1][0][2])
        downup[t] = np.real(result_me.states[t].ptrace(0)[2][0][1])
        anant1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[0][0][0])
        upupt1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[1][0][1])
        downdownt1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[2][0][2])
        updownt1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[1][0][2])
        downupt1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[2][0][1])
        upupbr[t] = np.real(result_br.states[t].ptrace(0)[0][0][0])
        downdownbr[t] = np.real(result_br.states[t].ptrace(0)[1][0][1])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][0])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[0][0][1])
        upupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[0][0][0])
        downdownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][1])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][0])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[0][0][1])

for noise_amplitude in np.linspace(0, 5, num=6):

    i = 1
    # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
    S = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(noise_amplitude, perturb_times, omega, bandwidth))

    # print('H0...')
    # print(H0(omega, J, N))
    # print('H1...')
    # print(H1(Omega_R, N))
    # print('H2...')
    # print(H2(Omega_R, N))

    result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], result_t1.states[timesteps - 1],
                      perturb_times, e_ops=Exps, options=opts)

    # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
    states2 = np.array(result2.states[timesteps - 1])
    expect2 = np.array(result2.expect[:])
    ancilla_overlap = []
    while i < 25:
        # print(i)
        i += 1
        # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
        S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                         noisy_func(noise_amplitude, perturb_times, omega, bandwidth))

        result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], result_t1.states[timesteps - 1],
                          perturb_times, e_ops=Exps, options=opts)

        ancilla_overlap.append(np.abs(productstateA(0, 1, N).dag() * np.array(result2.states[timesteps - 1])) ** 2)

        # print(ancilla_overlap)

        states2 += np.array(result2.states[timesteps - 1])
        expect2 += np.array(result2.expect[:])

    # print(ancilla_overlap)
    print(np.mean(ancilla_overlap))

    # func2 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
    # noisy_func2 = lambda t: func2(t + random_phase)
    noisy_data2 = noisy_func(noise_amplitude, perturb_times, omega, bandwidth)
    S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

    states2 = states2 / i
    expect2 = expect2 / i
    # print(Qobj(states2))
    # print((expect2[5]+expect2[8]).mean())
    density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
                           [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
    # print(density_matrix)
    result3 = mesolve(H0(omega, J, N), Qobj(states2), t2, [], e_ops=Exps, options=opts)

    # print('Initial state ....')
    # print(productstateZ(0, 0, N))
    # print(productstateZ(0, 0, N).dag()*sigmaz(1, N)*productstateZ(0, 0, N))

    # print('Commutator:', 1j * Commutator[0][0])
    # print('AntiCommutator: ', AntiCommutator[0][0])
    # print(np.correlate(S2(perturb_times), S2(perturb_times), "valid"))
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    freq = np.fft.fftfreq(perturb_times.shape[-1], d=1 / sampling_rate)
    ax[0, 0].plot(freq, np.abs(np.fft.fft(noisy_func(noise_amplitude, perturb_times, omega, bandwidth))), linestyle='',
                  marker='o', markersize='2', linewidth=0.0)
    # ax[0, 0].plot(freq, np.imag(np.fft.fft(noisy_func(noise_amplitude, perturb_times, bandwidth))), linestyle='--',
    # marker='o', markersize='5')

    # ax[0, 0].plot(freq, np.correlate(S2(perturb_times), S2(perturb_times), "valid")[0], linestyle='--', marker='o', markersize='5')
    # ax[0, 0].plot(perturb_times, func2(perturb_times))
    # ax[0, 0].plot(perturb_times, np.real(noisy_data2), 'o')
    # ax[0, 0].plot(perturb_times, np.real(S2(perturb_times)), lw=2)
    ax[0, 0].set_xlabel('F [MHz]')
    ax[0, 0].set_ylabel('Coupling Amplitude')
    # ax[0, 0].set_xlim([0, 0.4])

    ax[0, 1].plot(perturb_times, np.real(S2(perturb_times)), linestyle='-', marker='o', markersize='0', linewidth=1.0)
    ax[0, 1].set_xlabel('Time [us]')

    # random_amplitude = np.random.normal(0, noise_amplitude, size=len(perturb_times))
    # noisefreq = np.fft.fft(S2(perturb_times))
    # noisefreq[25:45] = envelope("Blackman", np.real(noisefreq[25:45]))+np.imag(noisefreq[25:45])
    # noisefreq[500-45:500-25] = envelope("Blackman", np.real(noisefreq[500-45:500-25])) + np.imag(noisefreq[500-45:500-25])
    # noisefreq[0:25] = 0
    # noisefreq[45:500-45] = 0
    # noisefreq[500-25:500] = 0
    # noisefreq = envelope("Blackman", np.fft.fft(S2(perturb_times)))
    # ax[1, 0].plot(freq, np.abs(noisefreq), label="Fequency after Window")

    # pulse = envelope("Blackman", np.fft.fftshift(S2(perturb_times)))
    # ax[1, 1].plot(perturb_times, np.fft.ifft(noisefreq), label="Frequency after shifted Window")
    # ax[1, 0].plot(t1, np.real(result_t1.expect[1]), label="MagnetizationZ")
    # ax[1, 0].plot(t1, np.real(result_t1.expect[2]), label="MagnetizationY")
    # ax[1, 0].plot(t1, np.real(result_t1.expect[0]), label="MagnetizationX")
    # ax[1, 0].plot(t1, result_t1.expect[3], label="tensor(SigmaZ,Id) ")
    # ax[1, 0].plot(t1, result_t1.expect[4], label="tensor(Id,SigmaZ) ")
    # ax[1, 0].set_xlabel('Free Evolution Time [us]')
    # ax[1, 0].set_ylabel('Magnetization')
    # ax[1, 0].legend(loc="upper right")
    # ax[1, 0].set_ylim([-1.1, 1.1])

    # ax[1, 1].plot(t2, np.real(result_AB.expect[1]), label="MagnetizationZ")
    # ax[1, 1].plot(t2, np.real(result_AB.expect[2]), label="MagnetizationY")
    # ax[1, 1].plot(t2, np.real(result_AB.expect[0]), label="MagnetizationX")
    # ax[1, 1].plot(t2, result_AB.expect[3], label="tensor(SigmaZ,Id)")
    # ax[1, 1].plot(t2, result_AB.expect[4], label="tensor(Id,SigmaZ)")
    # ax[1, 1].set_xlabel('After Perturbation Operator [us]')
    # ax[1, 1].legend(loc="right")
    # ax[1, 1].set_ylim([-1.1, 1.1])

    # ax[2, 0].plot(perturb_times, expect2[0], label="MagnetizationX")
    ax[1, 0].plot(perturb_times, np.real(expect2[1]), label="MagnetizationZ")
    # ax[2, 0].plot(perturb_times, expect2[2], label="MagnetizationY")
    # ax[2, 0].plot(perturb_times, expect2[3], label="tensor(SigmaZ,Id) ")
    # ax[2, 0].plot(perturb_times, expect2[4], label="tensor(Id,SigmaZ) ")
    # ax[2, 0].plot(perturb_times, np.real(expect2[5]), label="upup")
    # ax[2, 0].plot(perturb_times, np.real(expect2[6]), label="updown")
    # ax[2, 0].plot(perturb_times, np.real(expect2[7]), label="downup")
    # ax[2, 0].plot(perturb_times, np.real(expect2[8]), label="downdown")
    # ax[2, 0].plot(perturb_times, np.real(expect2[9]), label="aa")
    ax[1, 0].set_xlabel('Time Dependent Perturbation [us]')
    # ax[2, 0].legend(loc="right")
    # ax[2, 0].set_ylim([-1.1, 1.1])

    # ax[2, 1].plot(t2, result3.expect[0], label="MagnetizationX")
    # ax[1, 1].plot(t2, np.real(result3.expect[1]), label="MagnetizationZ")
    ax[1, 1].plot(perturb_times, np.real(S2(perturb_times)), linestyle='--', marker='o', markersize='3', linewidth=1.0)
    ax[1, 1].set_xlabel('Time [us]')
    # ax[2, 1].plot(t2, result3.expect[2], label="MagnetizationY")
    # ax[2, 1].plot(t2, result3.expect[3], label="tensor(SigmaZ,Id) ")
    # ax[2, 1].plot(t2, result3.expect[4], label="tensor(Id,SigmaZ) ")
    # ax[2, 1].plot(t2, np.real(result3.expect[5]), label="upup")
    # ax[2, 1].plot(t2, np.real(result3.expect[6]), label="updown")
    # ax[2, 1].plot(t2, np.real(result3.expect[7]), label="downup")
    # ax[2, 1].plot(t2, np.real(result3.expect[8]), label="downdown")
    # ax[2, 1].plot(t2, np.real(result3.expect[9]), label="aa")
    # ax[1, 1].set_xlabel('After Time Dependent Pertubation [us]')
    # ax[1, 1].legend(loc="right")
    ax[1, 1].set_xlim([0, 0.1])

    # ax[3, 0].plot(t2, result_AB_me.expect[0], label="MagnetizationX")
    ax[2, 0].plot(t2, np.real(result_me.expect[1]), label="MagnetizationZ")
    # ax[3, 0].plot(t2, result_AB_me.expect[2], label="MagnetizationY")
    ax[2, 0].plot(t2, upup, label="uu")
    ax[2, 0].plot(t2, updown, label="ud")
    ax[2, 0].plot(t2, downup, label="du")
    ax[2, 0].plot(t2, downdown, label="dd")
    ax[2, 0].plot(t2, anan, label="aa")
    # ax[3, 0].plot(t2, result_AB.expect[3], label="tensor(SigmaZ,Id)")
    # ax[3, 0].plot(t2, result_AB.expect[4], label="tensor(Id,SigmaZ)")
    ax[2, 0].set_xlabel('Lindblad Perturbation [1/J]')
    # ax[3, 0].legend(loc="right")
    # ax[3, 0].set_ylim([-1.1, 1.1])

    ax[2, 1].plot(perturb_times, np.real(S2(perturb_times)), linestyle='--', marker='o', markersize='3', linewidth=1.0)
    ax[2, 1].set_xlabel('Time [us]')
    # ax[3, 1].plot(t2, result_t1t2_me.expect[0], label="MagnetizationX")
    # ax[2, 1].plot(t2, np.real(result_t1t2_me.expect[1]), label="MagnetizationZ")
    # ax[3, 1].plot(t2, result_t1t2_me.expect[2], label="MagnetizationY")
    # ax[2, 1].plot(t2, upupt1t2me, label="upup")
    # ax[2, 1].plot(t2, updownt1t2me, label="updown")
    # ax[2, 1].plot(t2, downupt1t2me, label="downup")
    # ax[2, 1].plot(t2, downdownt1t2me, label="downdown")
    # ax[2, 1].plot(t2, anant1t2me, label="aa")
    # ax[3, 1].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
    # ax[3, 1].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
    # ax[2, 1].set_xlabel('After Lindblad Perturbation [us]')
    # ax[2, 1].legend(loc="right")
    ax[2, 1].set_xlim([0, 0.01])

    # ax[4, 0].plot(t2, result_t1t2_br.expect[0], label="MagnetizationX")
    # ax[4, 0].plot(t2, np.real(result_t1t2_br.expect[1]), label="MagnetizationZ")
    # ax[4, 0].plot(t2, result_t1t2_br.expect[2], label="MagnetizationY")
    # ax[4, 0].plot(t2, upupbr, label="upup")
    # ax[4, 0].plot(t2, updownbr, label="updown")
    # ax[4, 0].plot(t2, downupbr, label="downup")
    # ax[4, 0].plot(t2, downdownbr, label="downdown")
    # ax[4, 0].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
    # ax[4, 0].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
    # ax[4, 0].set_xlabel('Bloch Redfield Perturbation [1/J]')
    # ax[4, 0].legend(loc="right")
    # ax[4, 0].set_ylim([-1.1, 1.1])

    # ax[4, 1].plot(t2, result_t1t2_br.expect[0], label="MagnetizationX")
    # ax[4, 1].plot(t2, np.real(result_t1t2_br.expect[1]), label="MagnetizationZ")
    # ax[4, 1].plot(t2, result_t1t2_br.expect[2], label="MagnetizationY")
    # ax[4, 1].plot(t2, upupt1t2br, label="upup")
    # ax[4, 1].plot(t2, updownt1t2br, label="updown")
    # ax[4, 1].plot(t2, downupt1t2br, label="downup")
    # ax[4, 1].plot(t2, downdownt1t2br, label="downdown")
    # ax[4, 1].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
    # ax[4, 1].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
    # ax[4, 1].set_xlabel('After Bloch Redfield Pertubation [1/J]')
    # ax[4, 1].legend(loc="right")
    # ax[4, 1].set_ylim([-1.1, 1.1])
    fig.tight_layout()
    # plt.show()
    plt.savefig("Dephasing with Amplitude noise at" + str(np.round(noise_amplitude, 2)) + ".pdf")
