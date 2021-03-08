from qutip import *
import numpy as np
from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

N = 1

omega = 2 * np.pi * 35 * 10 ** 1  # MHz

Omega_R = 2 * np.pi * 5.6  # MHz

J = 0  # MHz

bandwidth = 10  # MHz

sampling_rate = 2 * np.pi * 65 * 10 ** 1  # MHz
endtime = 1
timesteps = int(endtime * sampling_rate)

gamma1 = 0  # MHz

pertubation_length = endtime / 1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

noise_amplitude = 0.000

perturb_times = np.linspace(0, pertubation_length, timesteps)
random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(noise_amplitude, perturb_times, omega, bandwidth))

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)

for noise_amplitude in np.linspace(0, 50, num=25):
    print("Noise Amplitude", noise_amplitude)
    for bandwidth in np.linspace(1, 9, num=1):
        #print("Bandwidth", bandwidth)
        i = 1
        # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
        S = Cubic_Spline(perturb_times[0], perturb_times[-1], brownian_func(noise_amplitude, perturb_times, omega, sampling_rate))

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

        while i < 50:
            #print(i)
            i += 1
            # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
            S = Cubic_Spline(perturb_times[0], perturb_times[-1],
                             brownian_func(noise_amplitude, perturb_times, omega, sampling_rate))

            result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], productstateZ(0, 0, N),
                              perturb_times, e_ops=Exps, options=opts)

            states2 += np.array(result2.states[timesteps - 1])
            expect2 += np.array(result2.expect[:])

        # func2 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
        # noisy_func2 = lambda t: func2(t + random_phase)
        noisy_data2 = brownian_func(noise_amplitude, perturb_times, omega, sampling_rate)
        S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

        states2 = states2 / i
        expect2 = expect2 / i
        # print(Qobj(states2))
        # print((expect2[5]+expect2[8]).mean())
        density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]],
                               [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
        # print(density_matrix)
        #result3 = mesolve(H0(omega, J, N), Qobj(states2), t2, [], e_ops=Exps, options=opts)

        # print('Initial state ....')
        # print(productstateZ(0, 0, N))
        # print(productstateZ(0, 0, N).dag()*sigmaz(1, N)*productstateZ(0, 0, N))

        # print('Commutator:', 1j * Commutator[0][0])
        # print('AntiCommutator: ', AntiCommutator[0][0])
        # print(np.correlate(S2(perturb_times), S2(perturb_times), "valid"))
        #result_me = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
        #                    result_t1.states[timesteps - 1],
        #                    perturb_times, [noise_amplitude * sigmap(1, 0, N) / 10, noise_amplitude * sigmam(1, 0, N) / 10], Exps,
        #                    options=opts)

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        freq = np.fft.fftfreq(perturb_times.shape[-1], d=1 / sampling_rate)
        fourier = np.abs(np.fft.fft(noisy_data2))

        max_pos = int(len(perturb_times) / 2) - np.argmax(fourier[int(len(perturb_times) / 2): len(perturb_times)])


        ax[0, 0].plot(freq[0:int(len(perturb_times)/2)], fourier[0:int(len(perturb_times)/2)], linestyle='',
                      marker='o', markersize='2', linewidth=0.0)
        ax[0, 0].plot(freq[0:int(len(perturb_times)/2)], lorentzian(freq, fourier[max_pos], max_pos,
                      noise_amplitude)[0:int(len(perturb_times)/2)], linestyle='-',
                      marker='o', markersize='0', linewidth=1.0)

        ax[0, 0].set_xlabel('F [MHz]')
        ax[0, 0].set_ylabel('Coupling Amplitude')

        ax[0, 1].plot(perturb_times, np.real(S2(perturb_times)), linestyle='-', marker='o', markersize='0', linewidth=1.0)
        ax[1, 1].set_xlim([0, 0.1])
        ax[0, 1].set_xlabel('Time [us]')

        ax[1, 0].plot(perturb_times, np.real(expect2[1]), label="MagnetizationZ")
        ax[1, 0].set_xlabel('Time Dependent Perturbation [us]')

        S = Cubic_Spline(perturb_times[0], perturb_times[-1], func(perturb_times, omega))

        result_me = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]],
                            productstateZ(0, 0, N),
                            perturb_times, [np.sqrt(noise_amplitude) * sigmap(0, N), np.sqrt(noise_amplitude) * sigmam(0, N)], Exps,
                            options=opts)

        expect_me = result_me.expect[:]

        ax[1, 0].plot(perturb_times, np.real(expect2[1]), label="MagnetizationZ")

        ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="MagnetizationZ")



        ax[1, 1].plot(perturb_times, np.real(S2(perturb_times)), linestyle='--', marker='o', markersize='3', linewidth=1.0)
        ax[1, 1].set_xlabel('Time [us]')
        ax[1, 1].set_xlim([0, 0.01])
        fig.tight_layout()
        # plt.show()
        plt.savefig("Phase Noise with gamma = %.4f MHz.png" %noise_amplitude)# and BW %.2f.pdf" % (noise_amplitude, bandwidth))
