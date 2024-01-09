from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

N = 10
averages=100
omega = 2 * np.pi * 10 ** (-10)  # MHz

Omega_R = 1 * np.pi   # MHz

gamma = 0*2/3 * np.pi  # MHz

J = 1 * np.pi / 10   # MHz

for gamma in np.logspace(-2, 1, num=20):
    print("gamma: ", gamma)
    for J in np.logspace(-2 , 1, num=20):
        print("J: ", J)

        bath = "markovian"

        endtime = 6

        timesteps = 100

        pertubation_length = endtime / 1

        t1 = np.linspace(0, endtime, timesteps)
        t2 = np.linspace(0, endtime, timesteps)

        perturb_times = np.linspace(0, pertubation_length, timesteps)

        Exps = [MagnetizationX(N), MagnetizationY(N), MagnetizationZ(N), sigmaz(0, N), sigmaz(1, N), sigmaz(N - 1, N),
                upup(0, N), upup(1, N), upup(N - 2, N), upup(N - 1, N),
                sigmap(0, N), sigmam(0, N), downdown(0, N)]

        opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)
        init_state = productstateZ(0, 1, N)
        #init_state = productstateX(0, 1, N)


        pertubation_length = endtime / 1
        perturb_times = np.linspace(0, pertubation_length, timesteps)

        #print('H0...')
        #print(H0(omega, J, N))
        #print('H1...')
        #print(H1(Omega_R, N))
        # print('H2...')
        # print(H2(Omega_R, N))


        noise = noisy_func(gamma, perturb_times, omega, bath)

        S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise)
        S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise))

        result1 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                          perturb_times, e_ops=Exps, options=opts)




        concmean = []
        # for t in range(0, timesteps):
        # concmean.append(concurrence(result2.states[t]))

        # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
        states1 = np.array(result1.states[timesteps - 1])
        expect1 = np.array(result1.expect[:])
        ancilla_overlap = []
        Smean = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
        Pmean = 0

        i = 1

        while i < 2:  # averages + int(2 * gamma):
            print(i)
            i += 1
            noise = noisy_func(gamma, perturb_times, omega, bath)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise)
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise))
            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            # data / 0.4)

            result1 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                              perturb_times, e_ops=Exps, options=opts)

            states1 += np.array(result1.states[timesteps - 1])
            expect1 += np.array(result1.expect[:])

            #Smean += np.abs(np.fft.fft(Omega_R * noise ** 2)) # /2/np.pi/timesteps
            #Pmean += np.abs(np.sum(Omega_R * noise ** 2)) # /timesteps
            # for t in range(0, timesteps):
            #    concmean[t] += concurrence(result2.states[t])

        # noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
        # S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

        states1 = states1 / i
        expect1 = expect1 / i
        Smean = Smean / i
        Pmean = Pmean / i
        concmean = np.array(concmean) / i

        # print(Qobj(states2))
        # print((expect2[5]+expect2[8]).mean())
        density_matrix = Qobj([[expect1[5][timesteps - 1], expect1[6][timesteps - 1]],
                               [expect1[7][timesteps - 1], expect1[8][timesteps - 1]]])
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

        # print(Pmean)

        fig, ax = plt.subplots(4, 2, figsize=(10, 10))

        ax[0, 1].plot(perturb_times, np.real(expect1[2]), color='#85bb65', label="mag_z")
        ax[0, 1].plot(perturb_times, np.real(expect1[0]), color='black', label="mag_x")
        # ax[0, 1].plot(perturb_times, (-0.25)*np.ones_like(perturb_times), color='black', linestyle="--")
        # ax[0, 1].plot(perturb_times, (0.25)*np.ones_like(perturb_times), color='black', linestyle="--")
        ax[0, 1].legend(loc="lower center")
        # ax[0, 1].plot(perturb_times, (0)*np.ones_like(perturb_times), color='black', linestyle="--")
        #ax[0, 1].set_xlabel('Time [1/Omega_Rabi]', fontsize=12)
        ax[0, 1].set_ylabel('Magnetization', fontsize=12)
        ax[0, 0].plot(perturb_times, np.real(expect1[6]), color='red', label="upup 1st_atom")
        ax[0, 0].plot(perturb_times, np.real(expect1[7]), color='blue', label="upup 2nd_atom")
        ax[0, 0].plot(perturb_times, np.real(expect1[8]), color='green', label="upup secondlast atom", linestyle="--")
        ax[0, 0].plot(perturb_times, np.real(expect1[9]), color='orange', label="upup last atom", linestyle="--")
        #ax[0, 0].set_xlabel('Time [1/Omega_Rabi]', fontsize=12)
        ax[0, 0].set_ylabel('Single', fontsize=12)
        # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
        ax[0, 0].legend(loc="lower center")
        # ax[0, 0].set_ylim([-0.501, -0.499])
        #plt.show()

















        #Omega_R = 0 * np.pi  # MHz

        #gamma = 0.01 * np.pi  # MHz

        #J = 1 * np.pi / N  # MHz

        noise0 = noisy_func(gamma, perturb_times, omega, bath)
        noise1 = noisy_func(gamma, perturb_times, omega, bath)
        noise2 = noisy_func(gamma, perturb_times, omega, bath)
        noise3 = noisy_func(gamma, perturb_times, omega, bath)
        noise4 = noisy_func(gamma, perturb_times, omega, bath)
        noise5 = noisy_func(gamma, perturb_times, omega, bath)
        noise6 = noisy_func(gamma, perturb_times, omega, bath)
        noise7 = noisy_func(gamma, perturb_times, omega, bath)
        noise8 = noisy_func(gamma, perturb_times, omega, bath)
        noise9 = noisy_func(gamma, perturb_times, omega, bath)

        S01 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise0)
        S02 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise0))
        S11 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise1)
        S12 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise1))
        S21 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise2)
        S22 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise2))
        S31 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise3)
        S32 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise3))
        S41 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise4)
        S42 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise4))
        S51 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise5)
        S52 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise5))
        S61 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise6)
        S62 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise6))
        S71 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise7)
        S72 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise7))
        S81 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise8)
        S82 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise8))
        S91 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise9)
        S92 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise9))

        result2 = mesolve([H0(omega, J, N), [-Omega_R * sigmap(0, N), S01], [-Omega_R * sigmam(0, N), S02],
                 [-Omega_R * sigmap(1, N), S11], [-Omega_R * sigmam(1, N), S12],
                 [-Omega_R * sigmap(2, N), S21], [-Omega_R * sigmam(2, N), S22],
                 [-Omega_R * sigmap(3, N), S31], [-Omega_R * sigmam(3, N), S32],
                 [-Omega_R * sigmap(4, N), S41], [-Omega_R * sigmam(4, N), S42],
                 [-Omega_R * sigmap(5, N), S51], [-Omega_R * sigmam(5, N), S52],
                 [-Omega_R * sigmap(6, N), S61], [-Omega_R * sigmam(6, N), S62],
                 [-Omega_R * sigmap(7, N), S71], [-Omega_R * sigmam(7, N), S72],
                 [-Omega_R * sigmap(8, N), S81], [-Omega_R * sigmam(8, N), S82],
                 [-Omega_R * sigmap(9, N), S91], [-Omega_R * sigmam(9, N), S92],

                 ], init_state,
                perturb_times, e_ops=Exps, options=opts)
        concmean = []
        # for t in range(0, timesteps):
        # concmean.append(concurrence(result2.states[t]))

        # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
        states2 = np.array(result2.states[timesteps - 1])
        expect2 = np.array(result2.expect[:])
        ancilla_overlap = []
        Smean = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
        Pmean = 0

        i = 1

        while i < 2:  # averages + int(2 * gamma):

            i += 1
            print(i)
            noise0 = noisy_func(gamma, perturb_times, omega, bath)
            noise1 = noisy_func(gamma, perturb_times, omega, bath)
            noise2 = noisy_func(gamma, perturb_times, omega, bath)
            noise3 = noisy_func(gamma, perturb_times, omega, bath)
            noise4 = noisy_func(gamma, perturb_times, omega, bath)
            noise5 = noisy_func(gamma, perturb_times, omega, bath)
            noise6 = noisy_func(gamma, perturb_times, omega, bath)
            noise7 = noisy_func(gamma, perturb_times, omega, bath)
            noise8 = noisy_func(gamma, perturb_times, omega, bath)
            noise9 = noisy_func(gamma, perturb_times, omega, bath)

            S01 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise0)
            S02 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise0))
            S11 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise1)
            S12 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise1))
            S21 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise2)
            S22 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise2))
            S31 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise3)
            S32 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise3))
            S41 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise4)
            S42 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise4))
            S51 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise5)
            S52 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise5))
            S61 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise6)
            S62 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise6))
            S71 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise7)
            S72 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise7))
            S81 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise8)
            S82 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise8))
            S91 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise9)
            S92 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise9))

            result2 = mesolve([H0(omega, J, N), [-Omega_R * sigmap(0, N), S01], [-Omega_R * sigmam(0, N), S02],
                     [-Omega_R * sigmap(1, N), S11], [-Omega_R * sigmam(1, N), S12],
                     [-Omega_R * sigmap(2, N), S21], [-Omega_R * sigmam(2, N), S22],
                     [-Omega_R * sigmap(3, N), S31], [-Omega_R * sigmam(3, N), S32],
                     [-Omega_R * sigmap(4, N), S41], [-Omega_R * sigmam(4, N), S42],
                     [-Omega_R * sigmap(5, N), S51], [-Omega_R * sigmam(5, N), S52],
                     [-Omega_R * sigmap(6, N), S61], [-Omega_R * sigmam(6, N), S62],
                     [-Omega_R * sigmap(7, N), S71], [-Omega_R * sigmam(7, N), S72],
                     [-Omega_R * sigmap(8, N), S81], [-Omega_R * sigmam(8, N), S82],
                     [-Omega_R * sigmap(9, N), S91], [-Omega_R * sigmam(9, N), S92],

                     ], init_state,
                    perturb_times, e_ops=Exps, options=opts)


            states2 += np.array(result2.states[timesteps - 1])
            expect2 += np.array(result2.expect[:])

            #Smean += np.abs(np.fft.fft(Omega_R * noise ** 2)) # /2/np.pi/timesteps
            #Pmean += np.abs(np.sum(Omega_R * noise ** 2)) # /timesteps
            # for t in range(0, timesteps):
            #    concmean[t] += concurrence(result2.states[t])

        # noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
        # S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

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

        # print(Pmean)

        def func1(x):

            noise0 = noisy_func(gamma, perturb_times, omega, bath)
            noise1 = noisy_func(gamma, perturb_times, omega, bath)
            noise2 = noisy_func(gamma, perturb_times, omega, bath)
            noise3 = noisy_func(gamma, perturb_times, omega, bath)
            noise4 = noisy_func(gamma, perturb_times, omega, bath)
            noise5 = noisy_func(gamma, perturb_times, omega, bath)
            noise6 = noisy_func(gamma, perturb_times, omega, bath)
            noise7 = noisy_func(gamma, perturb_times, omega, bath)
            noise8 = noisy_func(gamma, perturb_times, omega, bath)
            noise9 = noisy_func(gamma, perturb_times, omega, bath)

            S01 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise0)
            S02 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise0))
            S11 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise1)
            S12 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise1))
            S21 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise2)
            S22 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise2))
            S31 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise3)
            S32 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise3))
            S41 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise4)
            S42 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise4))
            S51 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise5)
            S52 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise5))
            S61 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise6)
            S62 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise6))
            S71 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise7)
            S72 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise7))
            S81 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise8)
            S82 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise8))
            S91 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise9)
            S92 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise9))

            a_result = mesolve([H0(omega, J, N), [-Omega_R * sigmap(0, N), S01], [-Omega_R * sigmam(0, N), S02],
                     [-Omega_R * sigmap(1, N), S11], [-Omega_R * sigmam(1, N), S12],
                     [-Omega_R * sigmap(2, N), S21], [-Omega_R * sigmam(2, N), S22],
                     [-Omega_R * sigmap(3, N), S31], [-Omega_R * sigmam(3, N), S32],
                     [-Omega_R * sigmap(4, N), S41], [-Omega_R * sigmam(4, N), S42],
                     [-Omega_R * sigmap(5, N), S51], [-Omega_R * sigmam(5, N), S52],
                     [-Omega_R * sigmap(6, N), S61], [-Omega_R * sigmam(6, N), S62],
                     [-Omega_R * sigmap(7, N), S71], [-Omega_R * sigmam(7, N), S72],
                     [-Omega_R * sigmap(8, N), S81], [-Omega_R * sigmam(8, N), S82],
                     [-Omega_R * sigmap(9, N), S91], [-Omega_R * sigmam(9, N), S92],

                     ], init_state,
                    perturb_times, e_ops=Exps, options=opts)


            return a_result.expect[:] , a_result.states[timesteps - 1]

        a_expects, a_states = parfor(func1, range(32*averages))

        np.save("ExpectsLocalBath" + bath + ", Omega_R =  %.2f, J =  %.2f,gamma = %.2f.npy" % (
                        Omega_R, J, gamma), a_expects)
        np.save("StatesLocalBath" + bath + ", Omega_R =  %.2f, J =  %.2f,gamma = %.2f.npy" % (
                        Omega_R, J, gamma), a_expects)

        mean=a_expects[0][2]
        for b in range(1,averages*32):
            mean+=a_expects[b][2]

        mean=mean/averages/32
        #print("a_expects:" , a_expects[0][2])

        #print("a_expects mean:" , mean)

        #ax[1, 1].plot(perturb_times,  a_expects[0][0], color='grey', linestyle="--", label="mag_x")
        ax[1, 1].plot(perturb_times,  mean, color='#85bb65', label="mag_z")
        #ax[1, 1].plot(perturb_times, np.real(expect2[2]), color='#85bb65', label="mag_z")
        #ax[1, 1].plot(perturb_times, np.real(expect2[0]), color='black', label="mag_x")
        # ax[1, 1].plot(perturb_times, (-0.25)*np.ones_like(perturb_times), color='black', linestyle="--")
        # ax[1, 1].plot(perturb_times, (0.25)*np.ones_like(perturb_times), color='black', linestyle="--")
        ax[1, 1].legend(loc="lower center")
        # ax[0, 1].plot(perturb_times, (0)*np.ones_like(perturb_times), color='black', linestyle="--")
        #ax[1, 1].set_xlabel('Time [1/Omega_Rabi]', fontsize=12)
        ax[1, 1].set_ylabel('Magnetization', fontsize=12)
        ax[1, 0].plot(perturb_times, np.real(expect2[6]), color='red', label="upup 1st_atom")
        ax[1, 0].plot(perturb_times, np.real(expect2[7]), color='blue', label="upup 2nd_atom")
        ax[1, 0].plot(perturb_times, np.real(expect2[8]), color='green', label="upup secondlast atom", linestyle="--")
        ax[1, 0].plot(perturb_times, np.real(expect2[9]), color='orange', label="upup last atom", linestyle="--")
        #ax[1, 0].set_xlabel('Time [1/Omega_Rabi]', fontsize=12)
        ax[1, 0].set_ylabel('Local Noise', fontsize=12)
        # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
        ax[1, 0].legend(loc="lower center")










        #Omega_R = 2 * np.pi  # MHz

        #gamma = 0.01 * np.pi  # MHz

        #J = 1 * np.pi / N  # MHz

        noise = noisy_func(gamma, perturb_times, omega, bath)

        S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise)
        S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise))

        result3 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                          perturb_times, e_ops=Exps, options=opts)
        concmean = []
        # for t in range(0, timesteps):
        # concmean.append(concurrence(result2.states[t]))

        # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
        states3 = np.array(result3.states[timesteps - 1])
        expect3 = np.array(result3.expect[:])
        ancilla_overlap = []
        Smean = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
        Pmean = 0

        i = 1

        while i < 2:  # averages + int(2 * gamma):
            print(i)
            i += 1
            noise = noisy_func(gamma, perturb_times, omega, bath)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise)
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise))
            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            # data / 0.4)

            result3 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                              perturb_times, e_ops=Exps, options=opts)

            states3 += np.array(result3.states[timesteps - 1])
            expect3 += np.array(result3.expect[:])

            #Smean += np.abs(np.fft.fft(Omega_R * noise ** 2)) # /2/np.pi/timesteps
            #Pmean += np.abs(np.sum(Omega_R * noise ** 2)) # /timesteps
            # for t in range(0, timesteps):
            #    concmean[t] += concurrence(result2.states[t])

        # noisy_data2 = noisy_func(gamma, perturb_times, omega, bath)
        # S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

        states3 = states3 / i
        expect3 = expect3 / i
        Smean = Smean / i
        Pmean = Pmean / i
        concmean = np.array(concmean) / i

        # print(Qobj(states2))
        # print((expect2[5]+expect2[8]).mean())
        density_matrix = Qobj([[expect3[5][timesteps - 1], expect3[6][timesteps - 1]],
                               [expect3[7][timesteps - 1], expect3[8][timesteps - 1]]])
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

        # print(Pmean)

        def func2(x):

            noise0 = noisy_func(gamma, perturb_times, omega, bath)

            S01 = Cubic_Spline(perturb_times[0], perturb_times[-1], noise0)
            S02 = Cubic_Spline(perturb_times[0], perturb_times[-1], np.conj(noise0))


            a_result = mesolve([H0(omega, J, N), [H1(Omega_R, N), S01], [H2(Omega_R, N), S02]], init_state,
                    perturb_times, e_ops=Exps, options=opts)


            return a_result.expect[:] , a_result.states[timesteps - 1]

        a_expects, a_states = parfor(func2, range(32*10))

        np.save("ExpectsGlobalBath" + bath + ", Omega_R =  %.2f, J =  %.2f,gamma = %.2f.npy" % (
                        Omega_R, J, gamma), a_expects)
        np.save("StatesGlobalBath" + bath + ", Omega_R =  %.2f, J =  %.2f,gamma = %.2f.npy" % (
                        Omega_R, J, gamma), a_expects)

        mean1=a_expects[0][2]
        for b in range(1,averages*32):
            mean1+=a_expects[b][2]

        mean1=mean1/averages/32
        #print("a_expects:" , a_expects[0][2])

        #print("a_expects mean:" , mean1)


        #ax[2, 1].plot(perturb_times, np.real(expect3[2]), color='#85bb65', label="mag_z")
        #ax[2, 1].plot(perturb_times, np.real(expect3[0]), color='black', label="mag_x")
        ax[2, 1].plot(perturb_times, mean1, color='#85bb65', label="mag_z", linestyle="--")
        #ax[2, 1].plot(perturb_times, np.real(expect3[0]), color='black', label="mag_x")
        # ax[1, 1].plot(perturb_times, (-0.25)*np.ones_like(perturb_times), color='black', linestyle="--")
        # ax[1, 1].plot(perturb_times, (0.25)*np.ones_like(perturb_times), color='black', linestyle="--")
        ax[2, 1].legend(loc="lower center")
        # ax[0, 1].plot(perturb_times, (0)*np.ones_like(perturb_times), color='black', linestyle="--")
        #ax[2, 1].set_xlabel('Time [1/Omega_Rabi]', fontsize=12)
        ax[2, 1].set_ylabel('Magnetization', fontsize=12)
        ax[2, 0].plot(perturb_times, np.real(expect3[6]), color='red', label="upup 1st_atom")
        ax[2, 0].plot(perturb_times, np.real(expect3[7]), color='blue', label="upup 2nd_atom")
        ax[2, 0].plot(perturb_times, np.real(expect3[8]), color='green', label="upup secondlast atom", linestyle="--")
        ax[2, 0].plot(perturb_times, np.real(expect3[9]), color='orange', label="upup last atom", linestyle="--")
        #ax[2, 0].set_xlabel('Time [1/Omega_Rabi]', fontsize=12)
        ax[2, 0].set_ylabel('Global Noise', fontsize=12)
        # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
        ax[2, 0].legend(loc="lower center")





        #ax[3, 1].plot(perturb_times, np.real(expect3[2])-np.real(expect2[2]), color='#85bb65', label="mag_z")
        ax[3, 1].plot(perturb_times, mean-mean1, color='#85bb65', label="mag_z_i_hope")
        #ax[3, 1].plot(perturb_times, np.real(expect3[0])-np.real(expect2[0]), color='black', label="mag_x")
        # ax[1, 1].plot(perturb_times, (-0.25)*np.ones_like(perturb_times), color='black', linestyle="--")
        # ax[1, 1].plot(perturb_times, (0.25)*np.ones_like(perturb_times), color='black', linestyle="--")
        ax[3, 1].legend(loc="lower center")
        # ax[0, 1].plot(perturb_times, (0)*np.ones_like(perturb_times), color='black', linestyle="--")
        ax[3, 1].set_xlabel('Time [2\pi/Omega_Rabi]', fontsize=12)
        ax[3, 1].set_ylabel('Diff Magnetization', fontsize=12)
        ax[3, 0].plot(perturb_times, np.real(expect3[6])-np.real(expect2[6]), color='red', label="upup 1st_atom")
        ax[3, 0].plot(perturb_times, np.real(expect3[7])-np.real(expect2[7]), color='blue', label="upup 2nd_atom")
        ax[3, 0].plot(perturb_times, np.real(expect3[8])-np.real(expect2[8]), color='green', label="upup secondlast atom", linestyle="--")
        ax[3, 0].plot(perturb_times, np.real(expect3[9])-np.real(expect2[9]), color='orange', label="upup last atom", linestyle="--")
        ax[3, 0].set_xlabel('Time [2\pi/Omega_Rabi]', fontsize=12)
        ax[3, 0].set_ylabel('Diff', fontsize=12)
        # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
        ax[3, 0].legend(loc="lower center")

        #print(np.real(expect3[6]))
        #plt.show()
        plt.savefig("bath" + bath + ", Omega_R =  %.2f, J =  %.2f,gamma = %.2f.pdf" % (
                        Omega_R, J, gamma))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))