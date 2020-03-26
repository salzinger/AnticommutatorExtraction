import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check


def productstateZ(up_atom, down_atom, N):
    basislist = np.full(N, basis(3, 2))
    basislist[up_atom] = basis(3, 0)
    basislist[down_atom] = basis(3, 1)
    # print(basislist)
    return tensor(basislist)


def productstateX(up_atom, down_atom, N):
    basislist = np.full(N, basis(3, 2))
    basislist[up_atom] = (basis(3, 0) + basis(3, 1)).unit()
    basislist[down_atom] = (basis(3, 0) + basis(3, 1)).unit()
    # print(basislist)
    return tensor(basislist)


def bellstate1(i, j, N):
    blist1 = np.full(N, basis(3, 2))
    blist2 = np.full(N, basis(3, 2))

    blist1[i] = basis(3, 0)
    blist1[j] = basis(3, 1)

    blist2[i] = basis(3, 1)
    blist2[j] = basis(3, 0)

    bell = tensor(blist1) + tensor(blist2)

    return bell.unit()


def bellstate2(i, j, N):
    blist1 = np.full(N, basis(3, 2))
    blist2 = np.full(N, basis(3, 2))

    blist1[i] = basis(3, 0)
    blist1[j] = basis(3, 1)

    blist2[i] = basis(3, 1)
    blist2[j] = basis(3, 0)

    bell = tensor(blist1) - tensor(blist2)

    return bell.unit()


def bellstate3(i, j, N):
    blist1 = np.full(N, basis(3, 2))
    blist2 = np.full(N, basis(3, 2))

    blist1[i] = basis(3, 1)
    blist1[j] = basis(3, 1)

    blist2[i] = basis(3, 0)
    blist2[j] = basis(3, 0)

    bell = tensor(blist1) - tensor(blist2)

    return bell.unit()


def bellstate4(i, j, N):
    blist1 = np.full(N, basis(3, 2))
    blist2 = np.full(N, basis(3, 2))

    blist1[i] = basis(3, 1)
    blist1[j] = basis(3, 1)

    blist2[i] = basis(3, 0)
    blist2[j] = basis(3, 0)

    bell = tensor(blist1) + tensor(blist2)

    return bell.unit()


def sigmaz(j, N):
    oplist = np.full(N, identity(3))
    oplist[j] = Qobj([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    return tensor(oplist)


def MagnetizationZ(N):
    sum = 0
    for j in range(0, N):
        sum += sigmaz(j, N)
    # print(sum/N)
    return sum / 2


def sigmax(j, N):
    oplist = np.full(N, identity(3))
    oplist[j] = Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    return tensor(oplist)


def MagnetizationX(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(j, N)
    # print(sum/N)
    return sum / 2


def disorder(E, N):
    oplist = []
    for n in range(0, N):
        oplist.append(
            E * np.random.random([1])[0] * basis(2, 0) * basis(2, 0).dag() + E * np.random.random([1])[0] * basis(2,
                                                                                                                  1) * basis(
                2, 1).dag())
    return tensor(oplist)


def upXY(j, N, C):
    oplist = []
    for n in range(0, N - 1):
        oplist.append(identity(3))
    oplist.insert(j, Qobj([[0, 0, 0], [C, 0, 0], [0, 0, 0]]))
    return tensor(oplist)


def downXY(j, N, C):
    oplist = []
    for n in range(0, N - 1):
        oplist.append(identity(3))
    oplist.insert(j, Qobj([[0, C, 0], [0, 0, 0], [0, 0, 0]]))

    return tensor(oplist)


def upD(j, N, Oc):
    oplist = []
    for n in range(0, N - 1):
        oplist.append(identity(3))
    oplist.insert(j, Qobj([[0, 0, 0], [0, 0, Oc], [0, 0, 0]]))
    return tensor(oplist)


def downD(j, N, Oc):
    oplist = []
    for n in range(0, N - 1):
        oplist.append(identity(3))
    oplist.insert(j, Qobj([[0, 0, 0], [0, 0, 0], [0, Oc, 0]]))
    return tensor(oplist)


def coupling(j, N, Oc):
    oplist = []
    for n in range(0, N - 1):
        oplist.append(identity(3))
    oplist.insert(j, Qobj([[0, 0, 0], [0, 0, Oc], [0, Oc, 0]]))
    return tensor(oplist)


def scatter(j, N, Gamma):
    oplist = []
    for n in range(0, N - 1):
        oplist.append(identity(3))
    oplist.insert(j, Qobj([[0, 0, 0], [0, 0, Gamma], [0, -Gamma, 0]]))
    return tensor(oplist)


def V(R, a, j, k):
    return (R / float((a * np.abs(k - j)))) ** 3


def J_eff(R, a, j, k):
    return 1 * V(R, a, j, k) / (2 + 2 * V(R, a, j, k) ** 2)


def G_eff(R, a, j, k):
    return 1 * V(R, a, j, k) ** 2 / (1 + V(R, a, j, k) ** 2) ** 2


def g_eff(R, a, j, k):
    return 1 * V(R, a, j, k) ** 4 / (1 + V(R, a, j, k) ** 2) ** 2


def H(R, N, C):
    HH = 0
    hh = 0
    for k in range(0, N):
        # H_e=H_e+coupling(k,N,1)
        for j in range(0, N):
            if k > j:
                Coupling = C  # * R / (np.abs(j - k))
                hh += 1
                HH += 1.0 * (
                        upXY(j, N, Coupling) * downXY(k, N, Coupling) + downXY(j, N, Coupling) * upXY(k, N, Coupling))

                # print("H", hh, ": Coherent XY dynamics between sites j=", j, "k=", k, "with Strength", Coupling)

    return HH


def H_eff(R, a, N):
    H_e = 0
    h_e = 0
    for k in range(0, N):
        # H_e=H_e+coupling(k,N,1)
        for j in range(0, N):
            if k > j:
                h_e + 1
                J = np.sqrt(J_eff(R, a, j, k))
                H_e += 0.5 * (upD(j, N, J) * downD(k, N, J) + upD(k, N, J) * downD(j, N, J))
                # print("H",h_e,": Exchange dynamics between sites k=",k,"j=",j,"at distance",a*np.abs(k-j),"with J_eff:","%6.5f" % J_eff(R,a,j,k),H_e)

    return H_e


scatterindex = []


def L_eff(R, a, N):
    L = []
    nlist = 0
    sitelist = []

    for j in range(0, N):
        L_j = 0
        # L.append(scatter(j,N,1))
        for k in range(0, N):
            if k > j:
                G = - np.sqrt(G_eff(R, a, j, k))
                L_j += (upD(k, N, G) * downD(j, N, G))
                scatterindex.append(k)
                # print("L",len(L)-1,": Transfer to scattering site k=",k,"from previous site j=",j," at distance=",a*np.abs(k-j),"with G_eff:","%6.5f" % G_eff(R,a,j,k))
                # L.append(upD(j, N, G) * downD(k, N, G))
                scatterindex.append(j)
                # print("L",len(L)-1,": Transfer to scattering site j=",j,"from previous site k=",k,"at distance=",a*np.abs(k-j),"with G_eff:","%6.5f" % G_eff(R,a,j,k))

            if k != j:
                g = 1j * np.sqrt(g_eff(R, a, j, k))
                L_j += (upD(k, N, g) * downD(k, N, g))
                scatterindex.append(j)
                # print("L",len(L)-1,": Scattering at j=",j,"due to impurity site k=",k,"at distance=",a*np.abs(k-j),"with g_eff:","%6.5f" % g_eff(R,a,j,k))
        L.append(L_j)
    # print(L)
    return L


N = 3
a = 1
timesteps = 100
ntraj = 200
statelist = []
opts = Options(store_states=True, store_final_state=True, ntraj=200)
'''
tracedEEav = np.zeros(timesteps)
tracedGGav = np.zeros(timesteps)
purityAav = np.zeros(timesteps)
purityBav = np.zeros(timesteps)
purityCav = np.zeros(timesteps)
concav = np.zeros(timesteps)
VNav = np.zeros(timesteps)

i = 1
start_up_atom = 0
start_down_atom = 3
read_out_atom = 3
radii = [0.5, 1, 1.5]
realtime = 11
'''
disavgs = 1


def call(N, radii, start_up_atom, start_down_atom, read_out_atom, realtime, start_state, perturb_duration, timesteps,
         opts):
    for r in radii:
        for t in np.ones(disavgs):
            # print(i, "of", disavgs)
            # i = i + 1

            overlap_op1 = bellstate1(start_up_atom, start_down_atom, N) * bellstate1(start_up_atom, start_down_atom,
                                                                                       N).dag()
            overlap_op2 = bellstate2(start_up_atom, start_down_atom, N) * bellstate2(start_up_atom, start_down_atom,
                                                                                       N).dag()
            overlap_op3 = bellstate3(start_up_atom, start_down_atom, N) * bellstate3(start_up_atom, start_down_atom,
                                                                                       N).dag()
            overlap_op4 = bellstate4(start_up_atom, start_down_atom, N) * bellstate4(start_up_atom, start_down_atom,
                                                                                       N).dag()

            ops = [MagnetizationX(N), overlap_op1, overlap_op2, overlap_op3, overlap_op4]
            print("Interaction Coefficient over one-half EIT bandwidth: ", "%6.3f" % (r ** (1 / 3)))
            print("Interparticle spacing in units of [Critical Radius R]: ", "%6.3f" % (a / r))
            print("Coherent hopping rate:", J_eff(r, a, 1, 2))
            print("Incoherent hopping rate:", G_eff(r, a, 1, 2))
            print("Incoherent scattering rate:", g_eff(r, a, 1, 2))
            print("Start up-atom:", start_up_atom, "Start down-atom:", start_down_atom, "  , Readout atom:",
                  read_out_atom)
            times = np.linspace(0.0, t * realtime, timesteps)
            perturb_times = np.linspace(0.0, t * perturb_duration, timesteps)
            # results = []
            # asklist = []
            # for j in range(0,N):
            # asklist.append(sigmaz(j,N))
            # result = mcsolve(H_eff(r,a,N) , productstate(0,N) , times, L_eff(r,a,N) , asklist, options=opts)
            result1 = mesolve(H(1, N, 1), start_state, times, [], ops, options=opts,
                              progress_bar=None)
            result2 = mesolve(H(1, N, 1) + H_eff(r, a, N), result1.states[timesteps - 1], perturb_times,
                              L_eff(r, a, N), ops,
                              options=opts,
                              progress_bar=True)
            result3 = mesolve(H(1, N, 1), result2.states[timesteps - 1], times, [],
                              ops, options=opts,
                              progress_bar=None)

            # print(result1.states[0].ptrace([start_up_atom, start_down_atom]).eigenenergies())
            # print(result1.states[99].ptrace([start_up_atom, start_down_atom]))
            # print(result2.states[99].ptrace([start_up_atom, start_down_atom]))
            # print(result3.states[99].ptrace([start_up_atom, start_down_atom]))

            '''
            trace=[]
            traceduu1 = []
            traceddd1 = []
            tracedbb1 = []
            purityA1 = []
            purityB1 = []
            purityC1 = []
            conc1 = []
            VN1 = []
            mag1x = []
            mag1z = []

            traceduu2 = []
            traceddd2 = []
            tracedbb2 = []
            purityA2 = []
            purityB2 = []
            purityC2 = []
            conc2 = []
            VN2 = []
            mag2x = []
            mag2z = []

            traceduu3 = []
            traceddd3 = []
            tracedbb3 = []
            purityA3 = []
            purityB3 = []
            purityC3 = []
            conc3 = []
            VN3 = []
            mag3x = []
            mag3z = []

            #print(result1.states[0])#.ptrace([start_up_atom, start_down_atom]).tr())
            #print(result2.states[99])#.ptrace([start_up_atom, start_down_atom]).tr())
            #print(result3.states[99])#.ptrace([start_up_atom, start_down_atom]).tr())

            for t in range(0, 0*timesteps):
                traceduu1.append(np.abs(result1.states[t].ptrace(read_out_atom)[0][0][0]))
                traceddd1.append(np.abs(result1.states[t].ptrace(read_out_atom)[1][0][1]))
                tracedbb1.append(np.abs(result1.states[t].ptrace(read_out_atom)[2][0][2]))
                # conc.append(concurrence(result.states[t]))
                VN1.append(entropy_vn(result1.states[t]))
                mag1x.append(result1.expect[0][t])
                mag1z.append(result1.expect[1][t])

                traceduu2.append(np.abs(result2.states[t].ptrace(read_out_atom)[0][0][0]))
                traceddd2.append(np.abs(result2.states[t].ptrace(read_out_atom)[1][0][1]))
                tracedbb2.append(np.abs(result2.states[t].ptrace(read_out_atom)[2][0][2]))
                trace.append(result2.states[t].tr())
                # conc2.append(concurrence(result.states2[t]))
                VN2.append(entropy_vn(result2.states[t]))
                mag2x.append(result2.expect[0][t])
                mag2z.append(result2.expect[1][t])

                traceduu3.append(np.abs(result3.states[t].ptrace(read_out_atom)[0][0][0]))
                traceddd3.append(np.abs(result3.states[t].ptrace(read_out_atom)[1][0][1]))
                tracedbb3.append(np.abs(result3.states[t].ptrace(read_out_atom)[2][0][2]))
                # conc.append(concurrence(result.states[t]))
                VN3.append(entropy_vn(result3.states[t]))
                mag3x.append(result3.expect[0][t])
                mag3z.append(result3.expect[1][t])
            
            '''
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            axs[0].plot(times, result1.expect[0][:])
            axs[0].plot(times, result1.expect[1][:])
            axs[0].plot(times, result1.expect[2][:])
            axs[0].plot(times, result1.expect[3][:])
            axs[0].plot(times, result1.expect[4][:])
            axs[0].set_xlabel('Time 1/J')
            axs[1].plot(perturb_times, result2.expect[0][:])
            axs[1].plot(perturb_times, result2.expect[1][:])
            axs[1].set_xlabel('Time 1/J')
            axs[2].plot(times, result3.expect[0][:], label="Magnetization x")
            axs[2].plot(times, result3.expect[1][:], label="Bell Overlap")
            axs[2].set_xlabel('Time 1/J')
            # axs[0].plot(times, result1.expect[1][:])
            # axs[1].plot(perturb_times, result2.expect[1][:])
            # axs[2].plot(times, result3.expect[1][:], label="Magnetization z")
            leg = plt.legend(loc='upper right', ncol=1, shadow=True, fancybox=False)
            leg.get_frame().set_alpha(0.5)

            plt.show()
            plt.gcf().clear()

            # rhoA=result.states[t].ptrace(0)
            # rhoAsquare=rhoA*rhoA
            # purityA.append(rhoAsquare[0][0][0]+rhoAsquare[1][0][1])

            # rhoB=result.states[t].ptrace(1)
            # rhoBsquare=rhoB*rhoB
            # purityB.append(rhoBsquare[0][0][0]+rhoBsquare[1][0][1])

            # rhoC=result.states[t].ptrace(1)
            # rhoCsquare=rhoC*rhoC
            # purityC.append(rhoCsquare[0][0][0]+rhoCsquare[1][0][1])

            # purityAav=np.add(purityAav,purityA)
            # purityBav=np.add(purityBav,purityB)
            # purityCav=np.add(purityCav,purityC)
            # concav=np.add(concav,conc)

            # print(bellstate(0,1,N))

            # print(H(r,N).eigenstates())

            # fig, ax = plt.subplots()

            # ax.plot(times, tracedEE, label="rho e1e1")
            # ax.plot(times, tracedGG, label="rho g1g1")
            # ax.plot(times, np.abs(purityA), label="PurityA")
            # ax.plot(times, np.abs(purityB), label="PurityB")
            # ax.plot(times, conc, label="Concurrence")
            # leg = plt.legend(loc='best', ncol=3, shadow=True, fancybox=True)
            # leg.get_frame().set_alpha(0.5)
            # plt.show()

            # for n in range(0,N):
            # results.append(result.expect[n])
            # ax.plot(times, result.expect[0], label="n=%d"%(0,))

            # leg = plt.legend(loc='best', ncol=3, shadow=True, fancybox=True)
            # leg.get_frame().set_alpha(0.5)

            # ax.set_xlabel('Time');
            # ax.set_ylabel('Expectation value sigma(z)');
            # plt.show()

            # plt.savefig("evolution at t="+str(t*10)+" R_crit"+str(r)[0]+"."+str(r)[2]+".pdf", dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None, metadata=None)
            # plt.gcf().clear()

            # statelist.append(Qobj(inpt=result.states, dims=[[len(result.states)], [1]], shape=[], type=None, isherm=None, fast=False, superrep=None))

            # scatterlist=[]

            # for o in range(0,ntraj):
            # for g in range(0,len(config.operlist[o])):
            # print("operator",config.operlist[o][g],"scattersite",scatterindex[config.operlist[o][g]])
            # scatterlist.append(scatterindex[config.operlist[o][g]])

            # plt.hist(scatterlist, bins=range(0,N+1),rwidth=0.5,align='right',weights=np.ones(len(scatterlist))/ntraj)
            # plt.title("Average scattered photons at time t="+str(t*10)+" with R_crit="+str(r))

            # plt.savefig("histogram at t= "+str(t*10)+" R_crit"+str(r)[0]+str(r)[2]+".pdf", dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None, metadata=None)

            # plt.gcf().clear()
            # fig, ax = plt.subplots()
    return
