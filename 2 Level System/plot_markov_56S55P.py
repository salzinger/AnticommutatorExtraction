import numpy as np

from Atoms import *
from Driving import *
import matplotlib.pyplot as plt


N = 1

#omega = 2 * np.pi * 21 * 10 ** 3  # MHz
#Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

#omega = 2 * np.pi * 21 * 10 ** (-20)  # MHz

omega = 0  # MHz
Omega_R = 2 * np.pi * 1 * 17.86 ** 0  # MHz



#Omega_R = 2 * np.pi * 12.3  # MHz

#gamma = 2 * np.pi * 15.0  # MHz

J = 0 * 10 ** 0  # MHz
#bath = 'Forward3MHzcsv.txt'
#data = np.loadtxt('Forward3MHzcsv.txt')
#timesteps = 2 * len(data)
#endtime = 0.2
#pertubation_length = endtime / 1

#perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)

#vars = np.array([np.zeros(6400), np.zeros(6400), np.zeros(6400)])
#means = np.array([np.zeros(6400), np.zeros(6400), np.zeros(6400)])
#n=0
timesteps = 100
endtime = 0.07*13.6
pertubation_length = endtime / 1
perturb_times = np.linspace(0, endtime, timesteps)
init_state = productstateZ(0, 0, N)

'''
for gamma in [3, 10, 30]:


            bath = "markovian"

            init_state = productstateZ(0, 0, N)

            data = np.loadtxt('Forward3MHzcsv.txt')
            timesteps = 100
            endtime = 0.2
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, endtime, len(data))

            print(len(perturb_times))

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath)))

            #print('H0...')
            #print(H0(omega, J, N))
            #print('H1...')
            #print(H1(Omega_R, N))
            #print('H2...')
            #print(H2(Omega_R, N))

            result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                              perturb_times, e_ops=Exps, options=opts)
            concmean = []
            # for t in range(0, timesteps):
            # concmean.append(concurrence(result2.states[t]))

            # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
            states2 = np.array(result2.states[timesteps - 1])
            expect2 = np.array(result2.expect[:])
            expect1 = np.array([result2.expect[:]])
            #print(expect2)
            ancilla_overlap = []
            #Smean = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
            #Pmean = 0
            i = 1
            while i < int(1):  # averages + int(2 * gamma):
                print(i)
                i += 1

                S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                  noisy_func(gamma, perturb_times, omega, bath))
                S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                  np.conj(noisy_func(gamma, perturb_times, omega, bath)))
                # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                # data / 0.4)

                result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                                  perturb_times, e_ops=Exps, options=opts)

                #qsave(result2, "i =  %.2f, gamma =  %.2f" % (i,gamma))

                states2 += np.array(result2.states[timesteps - 1])
                expect2 += np.array(result2.expect[:])
                expect1 = np.append(expect1, [np.array(result2.expect[:])], axis=0)
                #print(expect2)
                #print(np.var(expect1, axis=0)[1])

                #Smean += np.abs(
                #    np.fft.fft(Omega_R * noisy_func(gamma, perturb_times, omega, bath)) ** 2)  # /2/np.pi/timesteps
                #Pmean += np.abs(np.sum(Omega_R * noisy_func(gamma, perturb_times, omega, bath) ** 2))  # /timesteps
                # for t in range(0, timesteps):
                #    concmean[t] += concurrence(result2.states[t])

            # noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
            # S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

            #states2 = states2 / i
            expect2 = expect2 / i
            #print(expect2)
            #print(np.var(expect2))
            #Smean = Smean / i
            #Pmean = Pmean / i
            #concmean = np.array(concmean) / i
            vars[n] = np.var(expect1, axis=0)[1]
            means[n] = np.mean(expect1, axis=0)[1]

            n += 1
            #print(means)
            #print(vars)

            # print(Qobj(states2))
            # print((expect2[5]+expect2[8]).mean())
            #density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
            #                       [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
'''
################### MASTER EQUATION ############################################ 444444444444444444
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

with open('control_pulse.txt') as f:
    linesm0 = f.readlines()
with open('perturbation_pulse.txt') as f:
    linesm3 = f.readlines()


with open('control_error.txt') as f:
    linesm0e = f.readlines()
with open('perturbation_error.txt') as f:
    linesm3e = f.readlines()


# print(linesmf)
# print(linesm)
x0 = []
y0 = []
y0e = []
y3 = []
y3e = []
y5 = []
y5e = []
y15 = []
y15e = []
y30 = []
y30e = []

'''
3.953125
0.378901455848414
52.449438202247194
2.007774167644925
'''

for element in range(1, 22):
    x0.append(float(linesm0[element][0:5])*13.6)

    y0.append((float(linesm0[element][8:18])))
    y3.append((float(linesm3[element][8:18])))


    y0e.append((float(linesm0e[element][8:18])))
    y3e.append((float(linesm3e[element][8:18])))


print(len(x0))
print(len(y0))
print(x0)
print(y3)

#t1 = np.linspace(0, endtime, int(timesteps))

Omega_R = 2 * np.pi * 17.865/13.6 #MHz

S = Cubic_Spline(perturb_times[0], perturb_times[-1], func(perturb_times, omega))

result_m0 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    init_state,
                    perturb_times, [np.sqrt(1/60) * sigmaz(0, N), np.sqrt(1/60) * sigmaz(0, N)], Exps,
                    options=opts)

m0 = np.array(result_m0.expect[:])

result_m3 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    init_state,
                    perturb_times, [np.sqrt(40/13.6) * sigmaz(0, N), np.sqrt(40/13.6) * sigmaz(0, N)], Exps,
                    options=opts)



ax[0, 1].errorbar(x0, y0, y0e, marker="o", color='black', label=r'$\Omega_{Rp}=1.3\Omega_R, \gamma=\Omega_R/60$', linestyle='', markersize="4")

ax[0, 1].plot(perturb_times, np.real(m0[1]), color='black', linestyle='-')

#ax[1, 1].plot(perturb_times, np.ones_like(perturb_times)*0.5, color='grey', linestyle='--')
#ax[1, 1].plot(perturb_times, -np.ones_like(perturb_times)*0.5, color='grey', linestyle='--')
#ax[1, 1].plot(perturb_times, -np.ones_like(perturb_times)*0.0, color='grey', linestyle='--')

#ax[1, 1].plot(perturb_times, np.real(expect2[1]), color='#008b8b', label="Time Dependant")
#ax[1, 1].plot(perturb_times, means[0], color='g',
#             label="fbm 3", marker="s", markersize="0.2", linestyle="")
#ax[1, 1].plot(perturb_times, means[1], color='yellow',
#             label="fbm 10", marker="s", markersize="0.2", linestyle="")
#ax[1, 1].plot(perturb_times, means[2], color='r',
              #label="fbm 30", marker="s", markersize="0.2", linestyle="")
#ax[1, 1].plot(perturb_times, np.mean(expect1, axis=0)[1] + np.var(expect1, axis=0)[1], color='b',
#              label=" +std", marker="s", markersize="0.1", linestyle="")
#ax[1, 1].plot(perturb_times, np.mean(expect1, axis=0)[1] - np.var(expect1, axis=0)[1], color='r',
#              label="-std ", marker="s", markersize="0.1", linestyle="")

ysum3=[]
thsum3=[]
ysum=[]
for i in range(0,len(perturb_times) ):
    thsum3.append(np.sum(np.real(result_m3.expect[1][0:i])))
for i in range(0,len(y3) ):
    ysum3.append(np.sum(y3[0:i]))
    ysum.append(np.sum(y0[0:i]))


ax[0, 1].errorbar(x0, y3, y3e, marker="o", color='#85bb65',  label=r'$\Omega_{Rp}=1.3\Omega_R, \gamma=2.9\Omega_R$', linestyle='', markersize="4")
ax[0, 1].plot(perturb_times, np.real(result_m3.expect[1]), color='#85bb65', linestyle='-')

ax[1, 1].errorbar(x0, ysum3, y3e, marker="o", color='#85bb65',  label=r'$\Omega_{Rp}=1.3\Omega_R, \gamma=2.9\Omega_R$', linestyle='', markersize="4")

ax[1, 1].set_xlabel('Time [$1/\Omega_R$]', fontsize=16)
ax[1, 1].set_ylabel(r'$\Sigma_t  \langle S_z \rangle_t $', fontsize=16)
#ax[0, 1].errorbar(x0, ysum, y3e, marker="o", color='black',  label=r'$\Omega_{Rp}=1.3\Omega_R, \gamma=2.9\Omega_R$', linestyle='', markersize="4")
ax[1, 1].set_xlim([-0.005, .97])
#ax[0, 1].plot(perturb_times, thsum3, color='#85bb65', linestyle='-')


#ax[1, 1].set_ylim([-0.596, 0.596])
ax[0, 1].set_xlabel('Time [$1/\Omega_R$]', fontsize=16)
ax[0, 1].set_ylabel(r'$\langle S_z \rangle $', fontsize=16)
ax[0, 1].legend(loc="upper right", fontsize=12)
ax[0, 1].set_ylim([-0.6, 0.6])
ax[0, 1].set_xlim([-0.005, .97])

ax[0, 1].set_yticks(ticks=np.array([-0.5, -0.25, 0., 0.25, 0.5]))




fig.tight_layout()
plt.show()
#plt.savefig("gamma =  %.2f.png" % (gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))
