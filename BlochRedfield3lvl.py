import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.solver import Options, Result, config, _solver_safety_check

#opts = Options(store_states=True, store_final_state=True, ntraj=200)


def threebasis():
    return np.array([basis(3, 0), basis(3, 1), basis(3, 2)], dtype=object)


def productstateZ(up_atom, down_atom, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [down for _ in oplist]
    oplist[up_atom] = down
    oplist[down_atom] = down
    return tensor(oplist)


def productstateX(m, j, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [(up + down).unit() for _ in oplist]
    return tensor(oplist)


def anan(m,N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    oplist[m] = ancilla * ancilla.dag()
    return tensor(oplist)


def upup(m,N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    oplist[m] = up * up.dag()
    return tensor(oplist)


def downdown(m,N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    oplist[m] = down * down.dag()
    return tensor(oplist)

def sigmap(ancilla_coupling, m, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    if ancilla_coupling:
        oplist[m] = ancilla * up.dag()
    else:
        oplist[m] = up * down.dag()
    return tensor(oplist)


def sigmam(ancilla_coupling, m, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    if ancilla_coupling:
        oplist[m] = up * ancilla.dag()
    else:
        oplist[m] = down * up.dag()
    return tensor(oplist)


def sigmaz(ancilla_coupling, j, N):
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    if ancilla_coupling:
        oplist[j] = Qobj([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    else:
        oplist[j] = Qobj([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    return tensor(oplist)


def sigmax(ancilla_coupling, j, N):
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    if ancilla_coupling:
        oplist[j] = Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    else:
        oplist[j] = Qobj([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    return tensor(oplist)


def sigmay(ancilla_coupling, j, N):
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    if ancilla_coupling:
        oplist[j] = Qobj([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    else:
        oplist[j] = Qobj([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    return tensor(oplist)


def MagnetizationZ(N):
    sum = 0
    for j in range(0, N):
        sum += sigmaz(0, j, N)
    return sum / N


def MagnetizationX(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(0, j, N)
    return sum / N


def MagnetizationY(N):
    sum = 0
    for j in range(0, N):
        sum += sigmay(0, j, N)
    return sum / N


def H0(omega, J, N):
    H = 0
    for j in range(0, N):
        #H += 1 * omega / 2 * sigmaz(1, j, N)
        for i in range(0, N):
            if i != j:
                H += J * (sigmaz(1, i, N) * sigmaz(1, j, N))
    return H


def H1(Omega_p, Omega_c, N):
    H = 0
    for j in range(0, N):
        H -= Omega_c * (sigmap(1, j, N) + sigmam(1, j, N)) + Omega_p * (sigmap(0, j, N) + sigmam(0, j, N))
    return H


def noisy_func(noise_amplitude, perturb_times):
    random_phase = noise_amplitude * np.random.uniform(low=- np.pi, high=np.pi, size=perturb_times.shape[0]) + np.pi
    random_amplitude = np.random.uniform(low=0.8, high=1.8, size=perturb_times.shape[0])
    random_frequency = np.random.uniform(low=0.8, high=1.2, size=perturb_times.shape[0])

    for t in range(0, len(random_phase)-1):
        if divmod(t, np.random.randint(29, 30))[1] != 0:
            random_amplitude[t+1] = random_amplitude[t]
            random_phase[t + 1] = random_phase[t]
            random_frequency[t + 1] = random_frequency[t]

    func1 = lambda t: 0.5j*np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
    noisy_func1 = lambda t: random_amplitude * func1(t * random_frequency + random_phase)
    return noisy_func1(perturb_times)*0

def func(perturb_times):
    func1 = lambda t: 0.5j*np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
    return func1(perturb_times)*0


N = 1

omega = 2. * np.pi * 400

Omega_R = 2. * np.pi * 4

Omega_c = 2. * np.pi * 4
Omega_p = 2. * np.pi * 1

Gamma = 2. * np.pi * 6

J = 0

timesteps = 200

endtime = 1
pertubation_length = endtime/1

t1 = np.linspace(0, endtime, timesteps)
t2 = np.linspace(0, endtime, timesteps)

noise_amplitude = 0.000

perturb_times = np.linspace(0, pertubation_length, timesteps)
random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])

S1 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(noise_amplitude, perturb_times))

Exps = [MagnetizationX(N), MagnetizationZ(N), MagnetizationY(N), sigmaz(0, 0, N), sigmaz(0, N - 1, N), upup(0, N),
        sigmap(0, 0, N), sigmam(0, 0, N), downdown(0, N), anan(0, N)]

Commutatorlist = []
Anticommutatorlist = []

opts = Options(store_states=True, store_final_state=True)

result_t1 = mesolve(H1(Omega_p, Omega_c, N), productstateZ(0, N-1, N), t1, [], Exps, options=opts)

#print(productstateZ(0, N-1, N))

result_t1t2 = mesolve(H1(Omega_p, Omega_c, N), result_t1.states[timesteps - 1], t2, [], Exps, options=opts)

Perturb = MagnetizationX(N)
Measure = MagnetizationY(N)

#result_AB = mesolve(H1(Omega_p, Omega_c, N), Perturb * result_t1.states[timesteps - 1], t2, [], Exps, options=opts)
#print(H1(Omega_p, Omega_c, N))

gamma1 = 1

def ohmic_spectrum(w):
    if w == 0.0:  # dephasing inducing noise
        return gamma1
    else:  # relaxation inducing noise
        return gamma1  # / 2 * (w / (2 * np.pi)) * (w > 0.0)


R, ekets = bloch_redfield_tensor(H1(Omega_p, Omega_c, N), [[Gamma*sigmax(0, 0, N), ohmic_spectrum]])

print(R)

expt_list = bloch_redfield_solve(R, ekets, result_t1.states[timesteps-1], t2, Exps)

spectra_cb = [ohmic_spectrum]
ohmic = "{gamma1} / 2.0 * (w / (2 * pi)) * (w > 0.0)".format(gamma1=gamma1)
#, 0.0003*sigmam(1, 0, N)]

S = Cubic_Spline(perturb_times[0], perturb_times[-1], func(perturb_times))

result_brs = bloch_redfield_solve(R, ekets, result_t1.states[timesteps-1], perturb_times, Exps)


result_t1t2_brs = bloch_redfield_solve(R, ekets, result_t1.states[timesteps-1], t2, Exps)

result_br=brmesolve(H1(Omega_p, Omega_c, N), result_t1.states[timesteps-1], perturb_times, [], Exps, [], options=opts)

result_t1t2_br = mesolve(H1(Omega_p, Omega_c, N), result_br.states[timesteps - 1], t2, [], Exps, [], options=opts)


#result_me = mesolve([H1(Omega_p,Omega_c,N), [H1(Omega_p,Omega_c,N), S], [H1(Omega_p,Omega_c,N), S]], result_t1.states[timesteps - 1],
                    #perturb_times, [3*sigmap(1, 0, N), 3*sigmam(1, 0, N)], Exps, options=opts)

result_me = mesolve(H1(Omega_p,Omega_c,N), result_t1.states[timesteps - 1],
                    perturb_times, [Gamma*sigmam(0, 0, N), 0.0003*sigmam(1, 0, N)], Exps, options=opts)



result_t1t2_me = mesolve(H1(Omega_p,Omega_c,N), result_me.states[timesteps - 1], t2, [], Exps, options=opts)


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
ananbr = np.zeros(timesteps)
upupbr = np.zeros(timesteps)
downdownbr = np.zeros(timesteps)
updownbr = np.zeros(timesteps)
downupbr = np.zeros(timesteps)
anant1t2br = np.zeros(timesteps)
upupt1t2br = np.zeros(timesteps)
downdownt1t2br = np.zeros(timesteps)
updownt1t2br = np.zeros(timesteps)
downupt1t2br = np.zeros(timesteps)

if N == 1:
    for t in range(0, timesteps):
        anan[t] = np.real(result_me.states[t][0][0][0])
        upup[t] = np.real(result_me.states[t][1][0][1])
        downdown[t] = np.real(result_me.states[t][2][0][2])
        updown[t] = np.real(result_me.states[t][1][0][2])
        downup[t] = np.real(result_me.states[t][2][0][1])
        anant1t2me[t] = np.real(result_t1t2_me.states[t][0][0][0])
        upupt1t2me[t] = np.real(result_t1t2_me.states[t][1][0][1])
        downdownt1t2me[t] = np.real(result_t1t2_me.states[t][2][0][2])
        updownt1t2me[t] = np.real(result_t1t2_me.states[t][1][0][2])
        downupt1t2me[t] = np.real(result_t1t2_me.states[t][2][0][1])
        ananbr[t] = np.real(result_t1t2_me.states[t][0][0][0])
        upupbr[t] = np.real(result_br.states[t][1][0][1])
        downdownbr[t] = np.real(result_br.states[t][2][0][2])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][2])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t][2][0][1])
        anant1t2br[t] = np.real(result_t1t2_me.states[t][0][0][0])
        upupt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][1])
        downdownt1t2br[t] = np.real(result_t1t2_br.states[t][2][0][2])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][2])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t][2][0][1])
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
        ananbr[t] = np.real(result_t1t2_me.states[t].ptrace(0)[0][0][0])
        upupbr[t] = np.real(result_br.states[t].ptrace(0)[1][0][1])
        downdownbr[t] = np.real(result_br.states[t].ptrace(0)[2][0][2])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][2])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[2][0][1])
        anant1t2br[t] = np.real(result_t1t2_me.states[t][0][0][0])
        upupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][1])
        downdownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[2][0][2])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][2])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[2][0][1])


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            #ax[0, 0].plot(perturb_times, func2(perturb_times))
            #ax[0, 0].plot(perturb_times, noisy_data2, 'o')
            #ax[0, 0].plot(perturb_times, S2(perturb_times), lw=2)
            #ax[0, 0].set_xlabel('Time [1/J]')
            #ax[0, 0].set_ylabel('Coupling Amplitude')
            #ax[0, 0].set_xlim([0, 0.1])

            #ax[0, 1].plot(perturb_times, S2(perturb_times), lw=2)
            #ax[0, 1].set_xlabel('Time [1/J]')


            #ax[1, 0].plot(t1, np.real(result_t1.expect[1]), label="MagnetizationZ")
            #ax[1, 0].plot(t1, np.real(result_t1.expect[2]), label="MagnetizationY")
            #ax[1, 0].plot(t1, np.real(result_t1.expect[0]), label="MagnetizationX")
            #ax[1, 0].plot(t1, result_t1.expect[3], label="tensor(SigmaZ,Id) ")
            #ax[1, 0].plot(t1, result_t1.expect[4], label="tensor(Id,SigmaZ) ")
            #ax[1, 0].set_xlabel('Free Evolution Time [1/J]')
            #ax[1, 0].set_ylabel('Magnetization')
            #ax[1, 0].legend(loc="upper right")
            #ax[1, 0].set_ylim([-1.1, 1.1])


            #ax[1, 1].plot(t2, np.real(result_AB.expect[1]), label="MagnetizationZ")
            #ax[1, 1].plot(t2, np.real(result_AB.expect[2]), label="MagnetizationY")
            #ax[1, 1].plot(t2, np.real(result_AB.expect[0]), label="MagnetizationX")
            #ax[1, 1].plot(t2, result_AB.expect[3], label="tensor(SigmaZ,Id)")
            #ax[1, 1].plot(t2, result_AB.expect[4], label="tensor(Id,SigmaZ)")
            #ax[1, 1].set_xlabel('After Perturbation Operator [1/J]')
            #ax[1, 1].legend(loc="right")
            #ax[1, 1].set_ylim([-1.1, 1.1])

            #ax[2, 0].plot(perturb_times, expect2[0], label="MagnetizationX")
            #ax[2, 0].plot(perturb_times, expect2[1], label="MagnetizationZ")
            #ax[2, 0].plot(perturb_times, expect2[2], label="MagnetizationY")
            #ax[2, 0].plot(perturb_times, expect2[3], label="tensor(SigmaZ,Id) ")
            #ax[2, 0].plot(perturb_times, expect2[4], label="tensor(Id,SigmaZ) ")
#ax[2, 0].plot(perturb_times, expect2[5], label="upup")
#ax[2, 0].plot(perturb_times, expect2[6], label="updown")
#ax[2, 0].plot(perturb_times, expect2[7], label="downup")
#ax[2, 0].plot(perturb_times, expect2[8], label="downdown")
#ax[2, 0].plot(perturb_times, expect2[9], label="aa")
#ax[2, 0].set_xlabel('Time Dependent Perturbation [1/J]')
            #ax[2, 0].legend(loc="right")
            #ax[2, 0].set_ylim([-1.1, 1.1])

            #ax[2, 1].plot(t2, result3.expect[0], label="MagnetizationX")
            #ax[2, 1].plot(t2, np.real(result3.expect[1]), label="MagnetizationZ")
            #ax[2, 1].plot(t2, result3.expect[2], label="MagnetizationY")
            #ax[2, 1].plot(t2, result3.expect[3], label="tensor(SigmaZ,Id) ")
            #ax[2, 1].plot(t2, result3.expect[4], label="tensor(Id,SigmaZ) ")
#ax[2, 1].plot(t2, np.real(result3.expect[5]), label="upup")
#ax[2, 1].plot(t2, np.real(result3.expect[6]), label="updown")
#ax[2, 1].plot(t2, np.real(result3.expect[7]), label="downup")
#ax[2, 1].plot(t2, np.real(result3.expect[8]), label="downdown")
#ax[2, 1].plot(t2, np.real(result3.expect[9]), label="aa")
#ax[2, 1].set_xlabel('After Time Dependent Pertubation [1/J]')
#ax[2, 1].legend(loc="right")
            #ax[2, 1].set_ylim([-1.1, 1.1])

            #ax[3, 0].plot(t2, result_AB_me.expect[0], label="MagnetizationX")
            #ax[3, 0].plot(t2, np.real(result_me.expect[1]), label="MagnetizationZ")
            #ax[3, 0].plot(t2, result_AB_me.expect[2], label="MagnetizationY")
ax[0, 0].plot(t2, anan, label="rr")
ax[0, 0].plot(t2, upup, label="ee")
ax[0, 0].plot(t2, updown, label="eg")
ax[0, 0].plot(t2, downup, label="ge")
ax[0, 0].plot(t2, downdown, label="gg")

            #ax[3, 0].plot(t2, result_AB.expect[3], label="tensor(SigmaZ,Id)")
            #ax[3, 0].plot(t2, result_AB.expect[4], label="tensor(Id,SigmaZ)")
ax[0, 0].set_xlabel('Lindblad Perturbation [1/J]')
            #ax[3, 0].legend(loc="right")
            #ax[3, 0].set_ylim([-1.1, 1.1])

            #ax[3, 1].plot(t2, result_t1t2_me.expect[0], label="MagnetizationX")
            #ax[3, 1].plot(t2, np.real(result_t1t2_me.expect[1]), label="MagnetizationZ")
            #ax[3, 1].plot(t2, result_t1t2_me.expect[2], label="MagnetizationY")
ax[0, 1].plot(t2, anant1t2me, label="rr")
ax[0, 1].plot(t2, upupt1t2me, label="ee")
ax[0, 1].plot(t2, updownt1t2me, label="eg")
ax[0, 1].plot(t2, downupt1t2me, label="ge")
ax[0, 1].plot(t2, downdownt1t2me, label="gg")

            #ax[3, 1].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
            #ax[3, 1].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
ax[0, 1].set_xlabel('After Lindblad Perturbation [1/J]')
ax[0, 1].legend(loc="right")
            #ax[3, 1].set_ylim([-1.1, 1.1])

            #ax[4, 0].plot(t2, result_t1t2_br.expect[0], label="MagnetizationX")
            #ax[4, 0].plot(t2, np.real(result_t1t2_br.expect[1]), label="MagnetizationZ")
            #ax[4, 0].plot(t2, result_t1t2_br.expect[2], label="MagnetizationY")
ax[1, 0].plot(t2, ananbr, label="rr")
ax[1, 0].plot(t2, upupbr, label="ee")
ax[1, 0].plot(t2, updownbr, label="eg")
ax[1, 0].plot(t2, downupbr, label="ge")
ax[1, 0].plot(t2, downdownbr, label="gg")
            #ax[4, 0].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
            #ax[4, 0].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
ax[1, 0].set_xlabel('Bloch Redfield Perturbation [1/J]')
            #ax[4, 0].legend(loc="right")
            #ax[4, 0].set_ylim([-1.1, 1.1])

            #ax[4, 1].plot(t2, result_t1t2_br.expect[0], label="MagnetizationX")
#ax[4, 1].plot(t2, np.real(result_t1t2_br.expect[1]), label="MagnetizationZ")
            #ax[4, 1].plot(t2, result_t1t2_br.expect[2], label="MagnetizationY")

ax[1, 1].plot(t2, anant1t2br, label="rr")
ax[1, 1].plot(t2, upupt1t2br, label="ee")
ax[1, 1].plot(t2, updownt1t2br, label="eg")
ax[1, 1].plot(t2, downupt1t2br, label="ge")
ax[1, 1].plot(t2, downdownt1t2br, label="gg")
            #ax[4, 1].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
            #ax[4, 1].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
ax[1, 1].set_xlabel('After Bloch Redfield Pertubation [1/J]')
ax[1, 1].legend(loc="right")
            #ax[4, 1].set_ylim([-1.1, 1.1])
fig.tight_layout()
plt.show()
            #plt.savefig("Dephasing with"+str(noise_amplitude)+".pdf")
