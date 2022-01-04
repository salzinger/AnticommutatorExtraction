from Atoms import *
from Driving import *
import matplotlib.pyplot as plt


N = 1

#omega = 2 * np.pi * 21 * 10 ** 3  # MHz
#omega = 2 * np.pi * 21 * 10 ** (-20)  # MHz

omega = 0  # MHz

#Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 24.6 * 10 ** 0  # MHz
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

vars = np.array([np.zeros(6400), np.zeros(6400), np.zeros(6400)])
means = np.array([np.zeros(6400), np.zeros(6400), np.zeros(6400)])
n=0
for gamma in [3*np.pi*2, 10*np.pi*2, 30*np.pi*2]:


            bath = "markovian"

            init_state = productstateZ(0, 0, N)

            data = np.loadtxt('Forward3MHzcsv.txt')
            timesteps = 1 * len(data)
            endtime = 0.1
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
            while i < int(gamma/12):  # averages + int(2 * gamma):
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
            print(means)
            #print(vars)

            # print(Qobj(states2))
            # print((expect2[5]+expect2[8]).mean())
            #density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
            #                       [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])

################### MASTER EQUATION ############################################ 444444444444444444
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
with open('m0.txt') as f:
    linesm0 = f.readlines()
with open('m3.txt') as f:
    linesm3 = f.readlines()
with open('m10.txt') as f:
    linesm10 = f.readlines()
with open('m30.txt') as f:
    linesm30 = f.readlines()
with open('m0err.txt') as f:
    linesm0e = f.readlines()
with open('m3err.txt') as f:
    linesm3e = f.readlines()
with open('m10err.txt') as f:
    linesm10e = f.readlines()
with open('m30err.txt') as f:
    linesm30e = f.readlines()

# print(linesmf)
# print(linesm)
x0 = []
y0 = []
y0e = []
y3 = []
y3e = []
y10 = []
y10e = []
y30 = []
y30e = []

for element in range(1, 22):
    x0.append(float(linesm0[element][0:5]))
    y0.append(float(linesm0[element][8:18]))
    y3.append(float(linesm3[element][8:18]))
    y10.append(float(linesm10[element][8:18]))
    y30.append(float(linesm30[element][8:18]))
    y0e.append(float(linesm0e[element][8:18]))
    y3e.append(float(linesm3e[element][8:18]))
    y10e.append(float(linesm10e[element][8:18]))
    y30e.append(float(linesm30e[element][8:18]))

#t1 = np.linspace(0, endtime, int(timesteps))

#Omega_R = 2 * np.pi * 12.3 #MHz

S = Cubic_Spline(perturb_times[0], perturb_times[-1], func(perturb_times, omega))

result_m0 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    init_state,
                    perturb_times, [], Exps,
                    options=opts)

m0 = np.array(result_m0.expect[:])

result_m3 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                    init_state,
                    perturb_times, [np.sqrt(3) * sigmaz(0, N), np.sqrt(3) * sigmaz(0, N)], Exps,
                    options=opts)
result_m10 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                     init_state,
                     perturb_times, [np.sqrt(10) * sigmaz(0, N), np.sqrt(10) * sigmaz(0, N)], Exps,
                     options=opts)
result_m30 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                     init_state,
                     perturb_times, [np.sqrt(30) * sigmaz(0, N), np.sqrt(30) * sigmaz(0, N)], Exps,
                     options=opts)

ax[1, 1].errorbar(x0, y0, y0e, marker="o", color='black', label='$\gamma = 0$ MHz', linestyle='')

ax[1, 1].plot(perturb_times, np.real(m0[1]), color='black', linestyle='-')

#ax[1, 1].plot(perturb_times, np.real(expect2[1]), color='#008b8b', label="Time Dependant")
ax[1, 1].plot(perturb_times, means[0], color='g',
              label="fbm 3", marker="s", markersize="0.1", linestyle="")
ax[1, 1].plot(perturb_times, means[1], color='yellow',
              label="fbm 10", marker="s", markersize="0.1", linestyle="")
ax[1, 1].plot(perturb_times, means[2], color='r',
              label="fbm 30", marker="s", markersize="0.1", linestyle="")
#ax[1, 1].plot(perturb_times, np.mean(expect1, axis=0)[1] + np.var(expect1, axis=0)[1], color='b',
#              label=" +std", marker="s", markersize="0.1", linestyle="")
#ax[1, 1].plot(perturb_times, np.mean(expect1, axis=0)[1] - np.var(expect1, axis=0)[1], color='r',
#              label="-std ", marker="s", markersize="0.1", linestyle="")

ax[1, 1].errorbar(x0, y3, y3e, marker="^", color='#85bb65', label='$\gamma = 3$ MHz', linestyle='')
ax[1, 1].plot(perturb_times, np.real(result_m3.expect[1]), color='#85bb65', linestyle='-')
ax[1, 1].fill_between(perturb_times,  np.real(result_m3.expect[1])+vars[0],
                      np.real(result_m3.expect[1])-vars[0], alpha=0.2, color='#85bb65')

ax[1, 1].errorbar(x0, y10, y10e, marker="v", color='#CC7722', label='$\gamma = 10$ MHz', linestyle='')
ax[1, 1].plot(perturb_times, np.real(result_m10.expect[1]), color='#CC7722', linestyle='-')
ax[1, 1].fill_between(perturb_times,  np.real(result_m10.expect[1])+vars[1],
                      np.real(result_m10.expect[1])-vars[1], alpha=0.2, color='#CC7722')

ax[1, 1].errorbar(x0, y30, y30e, marker="s", color='#800020', label='$\gamma = 30$ MHz', linestyle='')
ax[1, 1].plot(perturb_times, np.real(result_m30.expect[1]), color='#800020', linestyle='-')
ax[1, 1].fill_between(perturb_times,  np.real(result_m30.expect[1])+vars[2],
                      np.real(result_m30.expect[1])-vars[2], alpha=0.2, color="#800020")

ax[1, 1].set_ylim([-0.596, 0.596])
ax[1, 1].set_xlabel('Time [$\mu$s]', fontsize=16)
ax[1, 1].set_ylabel('Magnetization', fontsize=16)
ax[1, 1].legend(loc="upper center", fontsize=12)

fig.tight_layout()
plt.show()
plt.savefig("gamma =  %.2f.png" % (gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))
