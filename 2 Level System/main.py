from qutip import *
import numpy as np
import pandas
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt
import array

def convert(s):
  # The function that converts the string to float
  s = s.strip().replace(',', '.')
  return float(s)

data = array.array('d') #an array of type double (float of 64 bits)

with open("noise_gamma_3.csv", 'r') as f:
    for l in f:
        strnumbers = l.split('\t')
        data.extend( (convert(s) for s in strnumbers if s!='') )
        #A generator expression here.

data = np.loadtxt('Forward3MHzcsv.txt')
#print(data)


N = 1

omega = 2 * np.pi * 21 * 10 ** 3  # MHz

Omega_R = 2 * np.pi * 24 * 10 ** 0  # MHz

gamma = 2 * np.pi * 15.0  # MHz

J = 0  # MHz

averages = 1

sampling_rate = 2 * np.pi * 64 * 10 ** 0  # MHz
endtime = 0.2
timesteps = int(endtime * sampling_rate)
timesteps = 2*len(data)

bath = 'Forward3MHzcsv.txt'

gamma1 = 0  # MHz

pertubation_length = endtime / 1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)#, nsteps=50000)




#print('done')
#data1 = file.read()
#print(data)
#print(len(data))

#data_reversed = -data[::-1]



#print(len(data_reversed))


#plt.clear()

#data = np.append(data, data_reversed)

#plt.plot(np.linspace(0, 0.2, int(len(data))), np.cumsum(data))
#plt.plot(np.linspace(0.1, 0.2, int(len(data))), np.cumsum(-data_reversed)+np.cumsum(data)[-1])
#plt.ylabel('Phase [°]')
#plt.xlabel('Time [us]')
#plt.legend()
#plt.show()


for o in np.logspace(np.log(15 * Omega_R), np.log(100 * Omega_R), num=3, base=np.e):
    print("omega: ", omega)
    for s in np.logspace(np.log(5 * omega), np.log(10 * omega), num=3, base=np.e):
        print("sampling: ", sampling_rate)
        init_state = productstateZ(0, 0, N)
        #timesteps = int(endtime * sampling_rate)
        timesteps = 2*len(data)
        endtime=0.2
        pertubation_length = endtime / 1
        t1 = np.linspace(0, endtime, timesteps)
        t2 = np.linspace(0, endtime, timesteps)
        perturb_times = np.linspace(0, pertubation_length, timesteps)
        print(len(perturb_times))
        for g in np.logspace(np.log(0.1 * Omega_R), np.log(10 * Omega_R), num=15, base=np.e):
            print("gamma: ", gamma)
            # print("Bandwidth", bandwidth)
            i = 1
            # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                             noisy_func(gamma, perturb_times, omega, bath))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noisy_func(gamma, perturb_times, omega, bath)))
            #S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            #                 data[0:32000]/0.4)

            #print('H0...')
            #print(H0(omega, J, N))
            #print('H1...')
            #print(H1(Omega_R, N))
            #print('H2...')
            #print(H2(Omega_R, N))

            result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                              perturb_times, e_ops=Exps, options=opts)
            concmean = []
            #for t in range(0, timesteps):
                #concmean.append(concurrence(result2.states[t]))

            # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
            states2 = np.array(result2.states[timesteps - 1])
            expect2 = np.array(result2.expect[:])
            ancilla_overlap = []
            Smean = np.zeros_like(perturb_times)+1j*np.zeros_like(perturb_times)
            Pmean=0

            while i < 1 :#averages + int(2 * gamma):
                # print(i)
                i += 1

                S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                  noisy_func(gamma, perturb_times, omega, bath))
                S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                  np.conj(noisy_func(gamma, perturb_times, omega, bath)))
                #S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                 #data / 0.4)

                result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                                  perturb_times, e_ops=Exps, options=opts)

                states2 += np.array(result2.states[timesteps - 1])
                expect2 += np.array(result2.expect[:])
                Smean += np.abs(np.fft.fft(Omega_R/2/np.pi*noisy_func(gamma, perturb_times, omega, bath))**2)
                Pmean += np.abs(np.sum(Omega_R*noisy_func(gamma, perturb_times, omega, bath) ** 2))/timesteps
                #for t in range(0, timesteps):
                #    concmean[t] += concurrence(result2.states[t])




            noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

            states2 = states2 / i
            expect2 = expect2 / i
            Smean = Smean / i
            Pmean = Pmean / i
            concmean = np.array(concmean) / i

            # print(Qobj(states2))
            # print((expect2[5]+expect2[8]).mean())
            density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
                                   [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
            # print(density_matrix)
            # result3 = mesolve(H0(omega, J, N), Qobj(states2), t2, [], e_ops=Exps, options=opts)

            # print('Initial state ....')
            # print(productstateZ(0, 0, N))
            # print(productstateZ(0, 0, N).dag()*sigmaz(1, N)*productstateZ(0, 0, N))

            # print('Commutator:', 1j * Commutator[0][0])
            # print('AntiCommutator: ', AntiCommutator[0][0])
            # print(np.correlate(S2(perturb_times), S2(perturb_times), "valid"))
            # result_me = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
            #                    result_t1.states[timesteps - 1],
            #                    perturb_times, [noise_amplitude * sigmap(1, 0, N) / 10, noise_amplitude * sigmam(1, 0, N) / 10], Exps,
            #                    options=opts)

            #print(Pmean)
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            freq = np.fft.fftfreq(perturb_times.shape[-1], d=1 / sampling_rate)
            fourier = Smean/timesteps**2#np.max(Smean) #np.abs(np.fft.fft(brownian_func(gamma, perturb_times, omega, sampling_rate)))

            max_pos = int(len(perturb_times) / 2) - np.argmax(fourier[int(len(perturb_times) / 2): len(perturb_times)])
            #print(len(perturb_times))
            #print(len(freq))
            #print(freq)
            #print(perturb_times)

            ax[0, 0].plot(freq[0:int(len(perturb_times)/2)], fourier[0:int(len(perturb_times)/2)], linestyle='',
                          marker='o', markersize='2', linewidth=0.0) #[0:int(len(perturb_times) / 2)]

            ax[0, 0].plot(freq[0:int(len(perturb_times) / 2)], lorentzian(freq, Pmean, omega/(2*np.pi),
                                                                          gamma)[0:int(len(perturb_times) / 2)],
                          linestyle='-',
                          marker='o', markersize='0', linewidth=1.0,
                          label="Lorentzian with FWHM gamma= %.2f MHz" % gamma)
            print(Pmean)
            print(np.sum(fourier[0:int(len(perturb_times))]))
            print(np.sum(lorentzian(freq, Pmean, omega / (2 * np.pi),
                       gamma)[0:int(len(perturb_times))]))

            ax[0, 0].legend(loc="upper right")

            ax[0, 0].set_xlabel('f [MHz]', fontsize=16)
            ax[0, 0].set_ylabel('Coupling Amplitude', fontsize=16)

            ax[0, 1].plot(perturb_times, np.real(S2(perturb_times)), linestyle='--', marker='o', markersize='3',
                          linewidth=1.0)
            ax[0, 1].set_xlabel('Time [us]', fontsize=16)
            ax[0, 1].set_ylabel('Coupling Amplitude', fontsize=16)
            ax[0, 1].set_xlim([0, 150 / omega])

            S = Cubic_Spline(perturb_times[0], perturb_times[-1], func(perturb_times, omega))


            result_me = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                                init_state,
                                perturb_times, [np.sqrt(gamma) * sigmaz(0, N)], Exps,
                                options=opts)

            expect_me = result_me.expect[:]

            ax[1, 0].plot(perturb_times, np.real(expect2[1]), label="sigma_z, Time Dependent Hamiltonian")
            #ax[1, 0].plot(perturb_times, np.real(expect2[0]), label="sigma_x, Time Dependent Hamiltonian")
            #ax[1, 0].plot(perturb_times, np.real(expect2[2]), label="sigma_y, Time Dependent Hamiltonian")
            ax[1, 0].plot(perturb_times, np.sqrt(expect2[2]**2+expect2[0]**2), label="xy-plane, Time Dependent Hamiltonian")
            #ax[1, 0].plot(perturb_times, concmean, label="overlap-bell-basis")
            #ax[1, 0].plot(perturb_times, np.exp(- perturb_times * gamma), color="orange", label="exp(- gamma * t)")
            #ax[1, 0].plot(perturb_times, -np.exp(- perturb_times * gamma), color="orange")
            ax[1, 0].set_xlabel('Time [us]', fontsize=16)
            ax[1, 0].set_ylabel('Magnetization', fontsize=16)
            #ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
            ax[1, 0].legend(loc="lower right")

            # Total time.
            T = perturb_times[-1]
            # Number of steps.
            Nsteps = len(perturb_times)
            # Time step size
            dt = T / Nsteps
            # Number of realizations to generate.
            m = averages
            # Create an empty array to store the realizations.
            x = np.empty((m, Nsteps + 1))
            # Initial values of x.
            x[:, 0] = 0

            phase_noise = brownian(x[:, 0], Nsteps, dt, np.sqrt(gamma), out=x[:, 1:])

            t = np.linspace(0.0, Nsteps * dt, Nsteps)

            for k in range(int(m / 10)):
                ax[1, 1].plot(t, phase_noise[k], color='grey', linewidth=0.1)

            ax[1, 1].plot(t, np.mean(phase_noise, axis=0), color='orange', linestyle='--', linewidth=2.0,
                          label='Real Mean')
            ax[1, 1].plot(t, np.sqrt(np.var(phase_noise, axis=0)), color='blue', linestyle='--', linewidth=2.0,
                          label='Real Standard Deviation')

            ax[1, 1].plot(t, np.sqrt(gamma * t), color='black', linestyle='--', linewidth=2.0,
                          label='Expected Standard Deviation = sqrt(gamma * t)')
            ax[1, 1].plot(t, -np.sqrt(gamma * t), color='black', linestyle='--', linewidth=2.0)
            ax[1, 1].set_ylim([-1.2 * np.sqrt(gamma * T), 1.2 * np.sqrt(gamma * T)])
            ax[1, 1].set_xlabel('Time [us]', fontsize=16)
            ax[1, 1].set_ylabel('Phase [pi/2]', fontsize=16)
            ax[1, 1].legend(loc="lower left")

            fig.tight_layout()
            plt.show()
            plt.savefig("bath" + bath + ", omega =  %.2f, sampling =  %.2f,gamma = %.2f.png" % (
            omega, sampling_rate, gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))
