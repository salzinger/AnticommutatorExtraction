from qutip import *
import numpy as np
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

N = 1

omega = 2 * np.pi * 35 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 5.0 * 10 ** 0  # MHz

gamma = 0.0  # MHz

J = 0  # MHz

averages = 100

sampling_rate = 2 * np.pi * 165 * 10 ** 0  # MHz
endtime = 5
timesteps = int(endtime * sampling_rate)

bath = "markovian"

gamma1 = 0  # MHz

pertubation_length = endtime / 1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)

for omega in np.logspace(np.log(5 * Omega_R), np.log(100 * Omega_R), num=3, base=np.e):
    print("omega: ", omega)
    for sampling_rate in np.logspace(np.log(5 * omega), np.log(10 * omega), num=3, base=np.e):
        print("sampling: ", sampling_rate)
        endtime = 5
        timesteps = int(endtime * sampling_rate)
        pertubation_length = endtime / 1
        t1 = np.linspace(0, endtime, timesteps)
        t2 = np.linspace(0, endtime, timesteps)
        perturb_times = np.linspace(0, pertubation_length, timesteps)
        for gamma in np.logspace(np.log(0.1 * Omega_R), np.log(10 * Omega_R), num=15, base=np.e):
            print("gamma: ", gamma)
            # print("Bandwidth", bandwidth)
            i = 1
            # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
            S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                             noisy_func(gamma, perturb_times, omega, bath))

            # print('H0...')
            # print(H0(omega, J, N))
            # print('H1...')
            # print(H1(Omega_R, N))
            # print('H2...')
            # print(H2(Omega_R, N))

            result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], productstateZ(0, 0, N),
                              perturb_times, e_ops=Exps, options=opts)

            # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
            states2 = np.array(result2.states[timesteps - 1])
            expect2 = np.array(result2.expect[:])
            ancilla_overlap = []
            Smean = np.zeros_like(perturb_times)+1j*np.zeros_like(perturb_times)

            while i < averages + int(2 * gamma):
                # print(i)
                i += 1

                S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                                 noisy_func(gamma, perturb_times, omega, bath))

                result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], productstateZ(0, 0, N),
                                  perturb_times, e_ops=Exps, options=opts)

                states2 += np.array(result2.states[timesteps - 1])
                expect2 += np.array(result2.expect[:])
                Smean += np.abs(np.fft.fft(noisy_func(gamma, perturb_times, omega, bath))**2)

            noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

            states2 = states2 / i
            expect2 = expect2 / i
            Smean = Smean / i
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

            ax[0, 0].plot(freq[0:int(len(perturb_times) / 2)], lorentzian(freq, 0.02, omega/(2*np.pi),
                                                                          gamma)[0:int(len(perturb_times) / 2)],
                          linestyle='-',
                          marker='o', markersize='0', linewidth=1.0,
                          label="Lorentzian with FWHM gamma= %.2f MHz" % gamma)

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
                                productstateZ(0, 0, N),
                                perturb_times, [np.sqrt(gamma) * sigmaz(0, N)], Exps,
                                options=opts)

            expect_me = result_me.expect[:]

            ax[1, 0].plot(perturb_times, np.real(expect2[1]), label="sigma_z, Time Dependent Hamiltonian")
            ax[1, 0].plot(perturb_times, np.exp(- perturb_times * gamma), color="orange", label="exp(- gamma * t)")
            ax[1, 0].plot(perturb_times, -np.exp(- perturb_times * gamma), color="orange")
            ax[1, 0].set_xlabel('Time [us]', fontsize=16)
            ax[1, 0].set_ylabel('Magnetization', fontsize=16)
            ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
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
            #plt.show()
            plt.savefig(bath + ", omega =  %.2f, sampling =  %.2f,gamma = %.2f.png" % (
            omega, sampling_rate, gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))
