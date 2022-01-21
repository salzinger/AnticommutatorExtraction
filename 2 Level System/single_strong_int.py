import numpy as np

from Atoms import *
from Driving import *
import matplotlib.pyplot as plt


N = 8

#omega = 2 * np.pi * 21 * 10 ** 3  # MHz
#omega = 2 * np.pi * 21 * 10 ** (-20)  # MHz

omega = 0  # MHz

#Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 15 * 10 ** 0  # MHz

gamma = 2 * np.pi * 15.0  # MHz

J = 2 * np.pi * 30 ** 0  # MHz

bath = '10MHz_gamma.txt'
data = np.loadtxt('10MHz_gamma.txt')
timesteps = 2 * len(data)
endtime = 0.4
pertubation_length = endtime / 1

perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)
figure = plt.plot()
c = Bloch(figure)
c.make_sphere()

for Omega_R in np.linspace(2*np.pi*15, 2*np.pi*15, 1):
    print("Omega_R: ", Omega_R)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for s in np.logspace(1 * omega, 10 * omega, num=1, base=np.e):
        # print("sampling: ", sampling_rate)
        init_state = productstateZ(0, 0, N)
        # timesteps = int(endtime * sampling_rate)
        data = np.loadtxt('10MHz_gamma.txt')
        timesteps = 2 * len(data)
        endtime = 0.4
        pertubation_length = endtime / 1
        # t1 = np.linspace(0, endtime, timesteps)
        # t2 = np.linspace(0, endtime, timesteps)
        perturb_times = np.linspace(0, pertubation_length, timesteps)
        fs = timesteps / endtime
        # print(len(perturb_times))
        for g in np.logspace(np.log(0.1 * Omega_R), np.log(10 * Omega_R), num=1, base=np.e):

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath)))
            '''
            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            # data / 0.4)
            #print(noisy_func(gamma, perturb_times, omega, bath))

            plt.plot(perturb_times, np.absolute(noisy_func(gamma, perturb_times, omega, bath))*2, label="Mag")
            plt.plot(perturb_times, np.angle(noisy_func(gamma, perturb_times, omega, bath)), label="Phase")
            plt.xlim(-0.001, 0.1)
            plt.ylim(-1, 2.25)
            plt.show()
            '''
            result_single = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
                                    perturb_times, e_ops=Exps, options=opts)

            expect_single = np.array(result_single.expect[:])
            qsave(result_single, "Omega_R =  %.2f" % Omega_R)
            # print("gamma: ", gamma)
            # print("Bandwidth", bandwidth)
            i = 1
            # random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

            bath = '10MHz_gamma.txt'

            gamma = 0
            data = np.loadtxt('10MHz_gamma.txt')
            timesteps = 2 * len(data)
            endtime = 0.4
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath)))
            # S = Cubic_Spline(perturb_times[0], perturb_times[-1],
            #                 data[0:32000]/0.4)

            #print('H0...')
            #print(H0(omega, J, N))
            #print('H1...')
            #print(H1(Omega_R, N))
            #print('H2...')
            #print(H2(Omega_R, N))

            #result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S2]], init_state,
            #                  perturb_times, e_ops=Exps, options=opts)
            concmean = []
            # for t in range(0, timesteps):
            # concmean.append(concurrence(result2.states[t]))

            # opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
            #states2 = np.array(result2.states[timesteps - 1])
            #expect2 = np.array(result2.expect[:])

            #################### SINGLE TRAJECTORY ######################################## 222222222222222222222222222


            # vec1 = [np.real(x1), np.real(y1), np.real(z1)]
            # vec2 = [np.real(x2), np.real(y2), np.real(z2)]
            # vec3 = [np.real(x3), np.real(y3), np.real(z3)]
            # th = np.linspace(0, 2*np.pi, 20)
            colorlist = []#np.zeros_like(data)
            data = np.cumsum(data)
            min = np.min(data)
            max = np.max(data)
            print(max)
            for element in data:
                if element > -88.5:
                    if element > 1.14:
                        colorlist.append([1-1*element/max, 1-40/80*element/max, 1-10/80*element/max, 1])
                    elif element < -1.14:
                        colorlist.append([1-10/80*element/min, 1-40/80*element/min, 1-1*element/min, element/min/4])
                    else:
                        colorlist.append([1, 0, 0, 1])
                elif element < -91.5:
                    colorlist.append([1-10/80*element/min, 1-40/80*element/min, 1-1*element/min, 1])
                else:
                    colorlist.append([0, 0, 0, 1])

            print(len(colorlist))
            print(len(expect_single[0]))

            expectsx = []
            expectsy = []
            expectsz = []
            colors = []

            for n in range(0, len(data)):
                if np.mod(n, 10) == 0:
                    expectsx.append(np.real(expect_single[0][n] * 2))
                    expectsy.append(np.real(expect_single[2][n] * 2))
                    expectsz.append(np.real(expect_single[1][n] * 2))
                    colors.append(colorlist[n])

            xz = np.real(expect_single[0]*2)
            yz = np.real(expect_single[2]*2)
            zz = np.real(expect_single[1]*2)
            c.point_color = colors
            c.point_size = [5, 20, 100]
            c.add_points([expectsx, expectsy, expectsz], 'm')
            #plt.colorbar(c.fig)
            c.sphere_alpha = 0.05
            # c.add_vectors(vec1)
            # c.add_vectors(vec2)
            # c.add_vectors(vec3)


            #c.clear()

            data = np.loadtxt('10MHz_gamma.txt')
            timesteps = 2 * len(data)
            endtime = 0.4
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            with open('counts_z_strong_int.txt') as f:
                linescountsz = f.readlines()

            with open('phase_fits_strong_int.txt') as f:
                linesphase = f.readlines()


            tmw = []
            z = []
            zerror = []
            amp = []
            amperror = []
            phase = []
            phaseerror = []
            y = []
            yerror = []
            xerror = []
            x = []
            total=[]

            Ntot=310
            de=10.6

            for element in range(2, 53):
                tmw.append(float(linescountsz[element][0:5]))

                z.append((float(linescountsz[element][7:15])-de)/(Ntot-de)-0.5)

                zerror.append(float(linescountsz[element][17:26])/(Ntot-de))

                amp.append(float(linesphase[element][15:25])/(Ntot-de))


                phase.append(float(linesphase[element][27:37])*2*np.pi/360)

                total.append(np.sqrt(  ( float(linesphase[element][15:25])/(Ntot-de) )**2
                                      +( (float(linesphase[element][7:15])-de)/(Ntot-de)  -  0.5 )**2))


            for element in range(56, 107):
                amperror.append(float(linesphase[element][9:17])/(Ntot-de))
                phaseerror.append(float(linesphase[element][22:31])*2*np.pi/360)

            #print(tmw)
            #print(z)
            #print(zerror)
            #print(phase)
            #print(amp)
            #print(amperror)
            #print(phaseerror)

            data = np.loadtxt('10MHz_gamma.txt')
            timesteps = 2 * len(data)
            endtime = 0.4
            pertubation_length = endtime / 1

            data_reversed = -data[::-1]

            data = np.cumsum(data)

            data_reversed = np.cumsum(data_reversed) + data[-1] + 180

            data = np.append(data / 180, data_reversed / 180)

            ax[0, 0].errorbar(perturb_times, data, label="Phase drift",
                              linewidth="0.4",
                              color='#85bb65')

            ax[0, 1].plot(perturb_times, np.real(expect_single[1]), color='#85bb65', linestyle="-")
            ax[0, 1].plot(perturb_times, np.sqrt(np.real(expect_single[0])**2+np.real(expect_single[2])**2), color='black', linestyle="-")
            ax[0, 1].errorbar(tmw, amp, amperror, color="black", label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}$", markersize="4", marker="s",
                         linestyle="")
            ax[0, 1].errorbar(tmw, z, zerror, color='#85bb65', label=r"$\langle \sigma_z \rangle}$", markersize="5", marker="o",
                         linestyle="")
            ax[0, 1].set_ylim([-0.68, 0.68])
            ax[0, 1].legend(loc="lower center", fontsize=12)



            ax[1, 0].errorbar(tmw, z, zerror, label=r"$\langle \sigma_z \rangle$", linestyle="", markersize="4", marker="o",
                          color='#85bb65')
            ax[1, 0].plot(perturb_times, np.real(expect_single[1]), color='#85bb65', linestyle="-")

            ax[1, 0].errorbar(tmw, amp, amperror, label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle}$",
                          linestyle="",
                          markersize="3", marker="s", color='black')

            ax[1, 0].plot(perturb_times, np.real(np.sqrt(expect_single[2] ** 2 + expect_single[0] ** 2)), color="black",
                          linestyle="--")

            ax[1, 0].set_xlabel(r'Time [$\mu$s]', fontsize=14)
            ax[1, 0].set_ylabel('Magnetization', fontsize=14)
            #ax[1, 0].set_ylim([-0.599, 0.599])
            # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
            ax[1, 0].legend(loc="lower center", fontsize=12)





            ax[1, 1].plot(perturb_times, -np.real(expect_single[0]), color='black', linestyle="-")
            ax[1, 1].plot(perturb_times, -np.real(expect_single[2]), color='#85bb65', linestyle="-")
            ax[1, 1].plot(perturb_times, np.sqrt(np.real(expect_single[1])**2+np.real(expect_single[0])**2+np.real(expect_single[2])**2), color='grey', linestyle="--", label="")
            ax[1, 1].errorbar(tmw, total, color="grey", markersize="4", marker="o", label=r"$\sqrt{\langle \sigma_z \rangle^2 + \langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}$",
                         linestyle="")

            ax[1, 1].errorbar(tmw, amp * np.cos(phase),  np.sqrt((np.array(amperror)*np.cos(np.array(phase)))**2+(np.array(amp)*np.sin(np.array(phase))*np.array(phaseerror))**2),
                                color='#85bb65', label=r"$\langle \sigma_x \rangle}$", markersize="4", marker="o", linestyle="")
            ax[1, 1].errorbar(tmw, amp * np.sin(phase),  np.sqrt((np.array(amperror)*np.sin(np.array(phase)))**2+(np.array(amp)*np.cos(np.array(phase))*np.array(phaseerror))**2),
                                color="black", label=r"$\langle \sigma_y \rangle}$", markersize="4", marker="s", linestyle="")
            ax[1, 1].set_ylim([-0.68, 0.68])
            ax[1, 1].legend(loc="lower center", fontsize=12)

            #plt.show()



            #plt.savefig("Omega_R =  %.2f.png" % (
            #    Omega_R))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))

c.render()
plt.show()

