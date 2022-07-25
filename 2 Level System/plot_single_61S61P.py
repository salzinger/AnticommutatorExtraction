import numpy as np

from Atoms import *
from Driving import *
import matplotlib.pyplot as plt

plt.rcParams.update({
  "text.usetex": True,

})


#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Latin Modern Roman']})

plt.rc('figure', figsize=(11.69, 8.27))


N = 1

#omega = 2 * np.pi * 21 * 10 ** 3  # MHz
#omega = 2 * np.pi * 21 * 10 ** (-20)  # MHz

omega = 0  # MHz

#Omega_R = 2 * np.pi * 25.7 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 14.6 * 10 ** 0  # MHz

Omega_R = 2 * np.pi * 15 * 10 ** 0  # MHz

gamma = 2 * np.pi * 15.0  # MHz

J = 0 * 10 ** 0  # MHz
bath = '10MHz_gamma.txt'
data = np.loadtxt('10MHz_gamma.txt')
timesteps = 2 * len(data)
endtime = 6
pertubation_length = endtime / 1

perturb_times = np.linspace(0, pertubation_length, timesteps)

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, N), sigmaz(N - 1, N), upup(0, N),
        sigmap(0, N), sigmam(0, N), downdown(0, N)]

opts = Options(store_states=True, store_final_state=True)  # , nsteps=50000)
figure = plt.plot()
c = Bloch(figure)
c.make_sphere()

Flist = []

for Omega_R in np.linspace(2*np.pi*1, 2*np.pi*1, 1, endpoint=1):
    #print("Omega_R: ", Omega_R)
    # print("sampling: ", sampling_rate)
    init_state = productstateZ(0, 0, N)
    print(Omega_R)
    # timesteps = int(endtime * sampling_rate)
    data = np.loadtxt('10MHz_gamma.txt')
    timesteps = 2 * len(data)
    endtime = 6
    pertubation_length = endtime / 1
    # t1 = np.linspace(0, endtime, timesteps)
    # t2 = np.linspace(0, endtime, timesteps)
    perturb_times = np.linspace(0, pertubation_length, timesteps)
    fs = timesteps / endtime
    # print(len(perturb_times))

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    for rise_time in np.linspace(700, 1000, 1, endpoint=1):

        rise_time = int(rise_time)

        print("rise_time= ", rise_time)


        for second_rise_time in np.linspace(900, 1500, 1, endpoint=1):

            g = Omega_R

            second_rise_time = int(second_rise_time)

            print("second_rise_time= ", second_rise_time)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time)))
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

            states = result_single.states

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
            endtime = 6
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            S1 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time))
            S2 = Cubic_Spline(perturb_times[0], perturb_times[-1],
                              np.conj(noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time)))
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
            #print(max)
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

            #print(len(colorlist))
            #print(len(expect_single[0]))

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
            endtime = 6
            pertubation_length = endtime / 1
            perturb_times = np.linspace(0, pertubation_length, timesteps)

            with open('counts_z_full_norm.txt') as f:
                linescountsz = f.readlines()

            with open('phase_fits_full_norm.txt') as f:
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
            total = []
            offset = []

            Ntot = 29.46153846153846
            de = 1.8333333333333333#-1.8

            Ntot = 24
            Ntot_std = 4

            Ntot = 38.416666666666664
            Ntot_std = 6.48

            de = 1.5

            #de = 1.5
            de_std=0.5

            #Ntot = 26.4
            #Ntot_std = 4

            #de = 1.5
            #de_std=0.5

            norm = []

            for element in range(2, 53):

                norm.append(2*float(linesphase[element][50:59])-de)

            mean_norm = np.mean(norm)

            print("Norm = ", mean_norm, "std_mean = ", np.sqrt(np.var(norm))/len(norm))


            #mean_norm = (24-de + mean_norm)/2

            #mean_norm = 24 - de

            for element in range(0, 51):

                norm[element] = mean_norm

            #print(2*np.mean(norm)-de)

            #norm = 2*np.mean(norm)-de



            for element in range(2, 53):
                tmw.append(float(linescountsz[element][0:5])*15)

                z.append(float(linescountsz[element][6:15]))

                zerror.append(float(linescountsz[element][16:25]))

                amp.append(float(linesphase[element][15:29]))

                #phase.append(float(linesphase[element][29:41])*2*np.pi/360+np.pi)

                phase.append(float(linesphase[element][29:38]) * 2 * np.pi / 360 + np.pi)

                offset.append(float(linesphase[element][50:59]))

                total.append( np.sqrt(  ( float(linesphase[element][15:29]) )**2  +  ( float(linescountsz[element][6:15]))**2 )  )

            for element in range(56, 107):
                amperror.append(float(linesphase[element][9:17]))
                try:
                    phaseerror.append(float(linesphase[element][20:30])*2*np.pi/360)
                except:
                    print("ERROR NAN")
                    phaseerror.append(200 * 2 * np.pi / 360)

            '''

            norm = []

            for element in range(2, 53):

                norm.append(2*float(linesphase[element][50:59])-de)

            mean_norm = np.mean(norm)

            print("Norm = ", mean_norm, "std_mean = ", np.sqrt(np.var(norm))/len(norm))


            #mean_norm = (24-de + mean_norm)/2

            #mean_norm = 24 - de

            for element in range(0, 51):

                norm[element] = mean_norm

            #print(2*np.mean(norm)-de)

            #norm = 2*np.mean(norm)-de



            for element in range(2, 53):
                tmw.append(float(linescountsz[element][0:5])*15)

                z.append((float(linescountsz[element][6:15])-de)/norm[element-2] - 0.5)

                zerror.append(float(linescountsz[element][16:25])/norm[element-2])

                amp.append(float(linesphase[element][15:29])/norm[element-2])

                #phase.append(float(linesphase[element][29:41])*2*np.pi/360+np.pi)

                phase.append(float(linesphase[element][29:38]) * 2 * np.pi / 360 + np.pi)

                offset.append(float(linesphase[element][50:59]))

                total.append( np.sqrt(  ( float(linesphase[element][15:29]) / norm[element-2] )**2  +  ( (float(linescountsz[element][6:15])-de) / norm[element-2]  -  0.5)**2 )  )

            for element in range(56, 107):
                amperror.append(float(linesphase[element][9:17])/(norm[element-2-56]))
                try:
                    phaseerror.append(float(linesphase[element][20:30])*2*np.pi/360)
                except:
                    print("ERROR NAN")
                    phaseerror.append(200 * 2 * np.pi / 360)

######################################################################


            norm = []

            for element in range(2, 53):

                norm.append(2*float(linesphase[element][46:54])-de)

            mean_norm = np.mean(norm)

            print(mean_norm)

            for element in range(0, 51):

                norm[element] = mean_norm


            for element in range(2, 53):
                tmw.append(float(linescountsz[element][0:5])*15)

                z.append((float(linescountsz[element][6:15])-de)/norm[element-2] - 0.5)

                zerror.append(float(linescountsz[element][16:25])/norm[element-2])

                amp.append(float(linesphase[element][15:26])/norm[element-2])

                #phase.append(float(linesphase[element][29:41])*2*np.pi/360+np.pi)

                phase.append(float(linesphase[element][26:36]) * 2 * np.pi / 360 + np.pi)

                offset.append(float(linesphase[element][50:59]))

                total.append( np.sqrt(  ( float(linesphase[element][15:26]) / norm[element-2] )**2  +  ( (float(linescountsz[element][6:15])-de) / norm[element-2]  -  0.5)**2 )  )


            for element in range(56, 107):
                amperror.append(float(linesphase[element][9:17])/(norm[element-2-56]))
                try:
                    phaseerror.append(float(linesphase[element][20:30])*2*np.pi/360)
                except:
                    print("ERROR NAN")
                    phaseerror.append(200 * 2 * np.pi / 360)

            '''


            #print("phases: ", phase)

            #print("z: ", z)

            #print("amps : ", amp)

            print("total : ", np.mean(total))

            print("sqrt(var) /len (total) : ", sqrt(np.var(total))/len(total))



            #print(tmw)
            #print(z)
            #print(zerror)

            #print(amp)
            #print(total)

            #print(amperror)
            #print(phase)
            #print(phaseerror)

            data = np.loadtxt('10MHz_gamma.txt')
            #len(data)
            timesteps = 2 * len(data)
            endtime = 6
            pertubation_length = endtime / 1

            data_reversed = -data[::-1]

            data = np.cumsum(data)

            data_reversed = np.cumsum(data_reversed) + data[-1] - 180

            data = np.append(data / 180, data_reversed / 180)

            #print(len(data))

            #print(len(noisy_func(gamma, perturb_times, omega, bath)))

            ax[0, 0].errorbar(perturb_times, data, label="Phase drift",
                              linewidth="0.4",
                              color='#85bb65')

            #ax[0, 0].errorbar(np.linspace(0, perturb_times[-1], len(noisy_func(gamma, perturb_times, omega, bath, rise_time))), -np.angle(noisy_func(gamma, perturb_times, omega, bath, rise_time))/np.pi, label="Phase drift filtered",
            #                  linewidth="0.4",
            #                  color='black')

            ax[0, 0].errorbar(np.linspace(0, perturb_times[-1], len(data)), np.abs(noisy_func(gamma, perturb_times, omega, bath, rise_time, second_rise_time)), label="Amp of Phase drift filtered",
                              linewidth="0.4",
                              color='black')


            ax[1, 0].plot(perturb_times, np.real(expect_single[1]), color='#85bb65', linestyle="-")
            ax[1, 0].plot(perturb_times, np.sqrt(np.real(expect_single[0])**2+np.real(expect_single[2])**2), color='black', linestyle="-")
            ax[1, 0].errorbar(tmw, amp, amperror, color="black", label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}/2$", markersize="4", marker="s",
                         linestyle="")
            ax[1, 0].errorbar(tmw, z, zerror, color='#85bb65', label=r"$\langle \sigma_z \rangle/2}$", markersize="5", marker="o",
                         linestyle="")
            ax[1, 0].set_ylim([-0.68, 0.68])
            ax[1, 0].legend(loc="lower center", fontsize=12)

            #print(len(tmw))
            #print(len(perturb_times) / 51)
            #discr=[]
            rho_measured = np.array([])
            rho_ideal = np.array([])

            F = []
            F1 = []
            F2 = []
            F3 = []
            F4 = []

            diz=[]
            dix=[]
            diy=[]

            for t in range(0, 51):
                #print(np.real(expect_single[1])[t*1004])
                #print(np.real(expect_single[1])[t*1004]-z[t])
                #discr.append((np.real(expect_single[1])[t*511])-z[t])

                np.append(rho_ideal, (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2).dag()/2)

                np.append(rho_measured, (qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2)

                F.append(np.sqrt(((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2).dag()/2

                                  * ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2).unit()).tr()))

                #F.append(np.sqrt( (  ((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2).dag()
                #          * (qeye(2) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2   ).tr()))

                #F.append( (( (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2).dag()
                #          * (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2).tr())

                #print((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1) + expect_single[2][t*511]*sigmay(0, 1) + expect_single[0][t*511]*sigmax(0, 1))/2)

                F1.append( np.sqrt((   qutip.to_choi(((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2))/2).dag()
                          * qutip.to_choi(  ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2 ).unit())   ).tr()))

                #F1.append( ( qutip.to_choi((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2).dag()
                #          * qutip.to_choi((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2)).tr())

                #print(states[t*511].dag()*states[t*511])

                F2.append((states[t*511].dag() * ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2).unit() * states[t*511] ).sqrtm())

                #F2.append( (states[t*511].dag() * (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2)
                #          * states[t*511])

                F3.append( fidelity(   (qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2,
                                    ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2).unit()    ))

                F4.append( process_fidelity((qeye(2) + expect_single[1][t*511]*sigmaz(0, 1)*2 + expect_single[2][t*511]*sigmay(0, 1)*2 + expect_single[0][t*511]*sigmax(0, 1)*2)/2,
                                            ((qeye(2)*(2*total[t]) + z[t]*sigmaz(0, 1)*2 + amp[t] * np.cos(phase[t])*sigmay(0, 1)*2 + amp[t] * np.sin(phase[t])*sigmax(0, 1)*2)/2).unit()    ))



                diz.append( -(np.real(expect_single[1])[t*511] - z[t]) / zerror[t] )

                dix.append( ((np.real(expect_single[2])[t*511]) - amp[t] * np.cos(phase[t])) / (np.sqrt((amperror[t]*np.cos(phase[t]))**2 + (amp[t]*np.sin(phase[t])*phaseerror[t])**2)) )

                diy.append( ((np.real(expect_single[0])[t*511]) - amp[t] * np.sin(phase[t])) / (np.sqrt((amperror[t]*np.sin(phase[t]))**2 + (amp[t]*np.cos(phase[t])*phaseerror[t])**2)) )

            #print(np.mean(F))
            #print(np.std(F))
            #print(np.mean(F1))
            #print(np.std(F1))
            #print(np.mean(F2))
            #print(np.std(F2))
            #print(np.mean(F3))
            #print(np.std(F3))

            Fmean = np.mean(F)
            Fmin = np.min(F)
            Fend = F[-1]

            print(Fmean)
            print(Fmin)

            Flist.append([np.round((Fmean-.9887002869959799)*100, decimals=3), np.round((Fmin-.9591676085073163)*100, decimals=3), np.round((Fend-.9967099830257092)*100 , decimals=3), rise_time, second_rise_time, Omega_R/(2*np.pi)])


            #print(np.std(F4))
            #print(np.sum(diz))
            #print(np.sum(dix))
            #print(np.sum(diy))

            #ax[1, 0].errorbar(tmw, diz, label=r"$z [\sigma]$", linestyle="--", markersize="4", marker="o",
             #             color='grey')


            #ax[1, 0].errorbar(tmw, F, label=r"$Fidelity $", linestyle="--", markersize="4", marker="o",
            #              color='black')

            ax[0, 1].errorbar(tmw, F, label=r"$F =\sqrt{ Tr[\rho^\dagger_{ideal} \rho_{measured}]}$", linestyle="--", markersize="4", marker="o",
                          color='black')

            ax[0, 1].errorbar(tmw, F2, label=r"$F =\sqrt{ \langle \Psi \vert \rho_{measured} \vert \Psi \rangle}$", linestyle="", markersize="3", marker="o",
                         color='blue')

            #ax[1, 0].errorbar(tmw, F3, label="Fidelity Qutip", linestyle="", markersize="3", marker="^",
            #             color='grey')



            ax[0, 1].errorbar(tmw, F1, label=r"$F_{process} = Tr[\chi^\dagger_{ideal}*\chi_{measured}]$", linestyle="--", markersize="4", marker="o",
                          color='g')

            #ax[1, 0].errorbar(tmw, F4, label="Process Fidelity Qutip", linestyle="", markersize="3", marker="s",
            #             color='orange')

            #ax[1, 0].errorbar(tmw, 1 - (np.abs( np.array(total)*2-1 ))/4 , #np.sqrt( np.array(amperror)**2 + np.array(zerror)**2),
            #                    color="grey", label=r"$1-\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2 + \langle \sigma_z \rangle^2}$", markersize="4", marker="s", linestyle="")


            #ax[1, 0].errorbar(tmw, dix, label=r"$y [\sigma]$", linestyle="--", markersize="4", marker="o",
            #              color='#85bb65')

            #ax[1, 0].errorbar(tmw, diy, label=r"$x [\sigma]$", linestyle="--", markersize="4", marker="o",
            #              color='black')

            #ax[1, 0].errorbar(tmw, z, zerror, label=r"$\langle \sigma_z \rangle$", linestyle="", markersize="4", marker="o",
            #              color='#85bb65')
            #ax[1, 0].plot(perturb_times, np.real(expect_single[1]), color='#85bb65', linestyle="-")

            #ax[1, 0].errorbar(tmw, amp, amperror, label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}/2$",
            #              linestyle="",
            #              markersize="3", marker="s", color='black')

            #ax[1, 0].plot(perturb_times, np.real(np.sqrt(expect_single[2] ** 2 + expect_single[0] ** 2)), color="black",
            #              linestyle="--")

            ax[0, 1].set_xlabel(r'Time [$\mu$s]', fontsize=14)
            ax[0, 1].set_ylabel(r'', fontsize=14)
            #ax[1, 0].set_ylim([-0.599, 0.599])
            # ax[1, 0].plot(perturb_times, np.real(expect_me[1]), label="sigma_z, ME with sqrt(gamma)*L")
            ax[0, 1].legend(loc="lower left", fontsize=12)

            ax[1, 0].plot(perturb_times, np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')


            ax[0, 1].plot(perturb_times, np.ones_like(perturb_times) * 1, color='grey', linestyle='--')
            #ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 1, color='grey', linestyle='--')
            #ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')

            ax[1, 1].plot(perturb_times, np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 1].plot(perturb_times, -np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 1].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')





            #ax[1, 1].errorbar(tmw, (1-np.array(F1))*10, label=r"$(1-F_{process})*10$", linestyle="--", markersize="4", marker="o",
            #              color='g')
            ax[1, 1].plot(perturb_times, -np.real(expect_single[0]), color='blue', linestyle="-")
            ax[1, 1].plot(perturb_times, -np.real(expect_single[2]), color='purple', linestyle="-")
            #ax[1, 1].plot(perturb_times, np.sqrt(np.real(expect_single[1])**2+np.real(expect_single[0])**2+np.real(expect_single[2])**2), color='grey', linestyle="--", label="")
            #ax[1, 1].errorbar(tmw, total, color="grey", markersize="4", marker="o", label=r"$\sqrt{\langle \sigma_z \rangle^2 + \langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2}$",
            #             linestyle="")
            ax[1, 1].errorbar(tmw, total, np.sqrt( np.array(amperror)**2 + np.array(zerror)**2),
                                color="grey", label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2 + \langle \sigma_z \rangle^2}/2$", markersize="4", marker="s", linestyle="")
            ax[1, 1].errorbar(tmw, -np.array(amp * np.cos(phase)),  np.sqrt((np.array(amperror)*np.cos(np.array(phase)))**2+(np.array(amp)*np.sin(np.array(phase))*np.array(phaseerror))**2),
                                color='purple', label=r"$\langle \sigma_y \rangle/2}$", markersize="4", marker="o", linestyle="")
            ax[1, 1].errorbar(tmw, -np.array(amp * np.sin(phase)),  np.sqrt((np.array(amperror)*np.sin(np.array(phase)))**2+(np.array(amp)*np.cos(np.array(phase))*np.array(phaseerror))**2),
                                color="blue", label=r"$\langle \sigma_x \rangle/2}$", markersize="4", marker="s", linestyle="")




            ax[1, 0].plot(perturb_times, np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
            ax[1, 0].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')

            ax[1, 1].set_ylim([-0.68, 0.68])
            ax[1, 1].legend(loc="lower center", fontsize=12)

            ax[1, 0].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=16)
            ax[1, 0].set_ylabel('', fontsize=14)
            ax[1, 0].set_ylabel('Fidelity', fontsize=14)
            ax[1, 1].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=16)
            ax[1, 1].set_ylabel('Magnetization', fontsize=16)

            ax[0, 1].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=16)
            ax[1, 0].set_ylabel('Magnetization', fontsize=16)

            print(Flist)
            #plt.show()

            #plt.yticks(np.arange(0, 6, 0.25))
            #plt.xticks(np.linspace(0, 6, 12))
            #plt.axis('scaled')

            #plt.savefig("Omega_R =  %.2f.png" % (
            #    Omega_R))  # and BW %.2f.pdf" % (noise_amplitude, bandwidth))

print(Flist)

c.render()
plt.show()


fig, ax = plt.subplots(3, 1, figsize=(10, 10))

ax[0].errorbar(perturb_times, data,
                  linewidth="0.4",
                  color='#85bb65')

#ax[0].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=14)
ax[0].set_ylabel(r'$\Phi_B(t)[\pi]$', fontsize=18)
#ax[0].legend(loc="lower center", fontsize=12)

#ax[0].set_ylim([-0.6, 0.75])
ax[0].set_xlim([0, 6.051])

#ax[0].errorbar(tmw, total , np.sqrt( np.array(amperror)**2 + np.array(zerror)**2),
 #                   color="grey", label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2 + \langle \sigma_z \rangle^2}/2$", markersize="4", marker="s", linestyle="")

ax[1].errorbar(tmw, -np.array(amp * np.sin(phase)),  np.sqrt((np.array(amperror)*np.sin(np.array(phase)))**2+(np.array(amp)*np.cos(np.array(phase))*np.array(phaseerror))**2),
                    color='black', label=r"$\langle S_x \rangle$", markersize="4", marker="s", linestyle="")

ax[1].errorbar(tmw, -np.array(amp * np.cos(phase)),  np.sqrt((np.array(amperror)*np.cos(np.array(phase)))**2+(np.array(amp)*np.sin(np.array(phase))*np.array(phaseerror))**2),
                    color='#800080', label=r"$\langle S_y \rangle$", markersize="4", marker="o", linestyle="")

ax[1].errorbar(tmw, z, zerror, color='#85bb65', label=r"$\langle S_z \rangle$", markersize="5", marker="o",
                  linestyle="")

ax[1].plot(perturb_times, -np.real(expect_single[0]), color='black', linestyle="-")
ax[1].plot(perturb_times, -np.real(expect_single[2]), color='#800080', linestyle="-")
ax[1].plot(perturb_times, np.real(expect_single[1]), color='#85bb65', linestyle="-")




#ax[0].errorbar(tmw, total, np.sqrt(np.array(amperror) ** 2 + np.array(zerror) ** 2),
#                  color="grey",
#                  #label=r"$\sqrt{\langle \sigma_x \rangle^2 + \langle \sigma_y \rangle^2 + \langle \sigma_z \rangle^2}/2$",
#                  label=r"Total",
#                  markersize="4", marker="s", linestyle="")

ax[1].plot(perturb_times, np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
ax[1].plot(perturb_times, -np.ones_like(perturb_times) * 0.5, color='grey', linestyle='--')
ax[1].plot(perturb_times, -np.ones_like(perturb_times) * 0.0, color='grey', linestyle='--')

ax[1].set_ylim([-0.6, 0.75])
ax[1].set_xlim([0, 6.051])
ax[1].set_yticks(ticks=np.array([-0.5, -0.25, 0., 0.25, 0.5]))




#ax[0].set_ylim([-0.68, 0.68])
ax[1].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=18)
ax[1].set_ylabel(r'Spin $\langle S_i \rangle$', fontsize=18)
ax[1].legend(loc="lower center", fontsize=14)




ax[2].errorbar(tmw, F2,
                  label=r"$F =\sqrt{ \langle \Psi \vert \rho_{measured} \vert \Psi \rangle}$",
                  linestyle="dotted", markersize="3", marker="o",
                  color='black')

ax[2].set_xlabel(r'Time [$1/\Omega_R$]', fontsize=18)
ax[2].set_ylabel('Fidelity', fontsize=18)
ax[2].set_xlim([0, 6.051])
#ax[2].legend(loc="lower left", fontsize=12)



plt.show()






