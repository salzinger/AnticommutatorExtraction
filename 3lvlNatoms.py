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
    oplist[up_atom] = up
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
        H += 1 * omega / 2 * sigmaz(1, j, N)
        for i in range(0, N):
            if i != j:
                H += J * (sigmax(0, i, N) * sigmax(0, j, N) + sigmay(0, i, N) * sigmay(0, j, N))
    return H


def H1(Omega_R, N):
    H = 0
    for j in range(0, N):
        H -= Omega_R * (sigmap(1, j, N))
    return H


def H2(Omega_R, N):
    H = 0
    for j in range(0, N):
        H -= Omega_R * (sigmam(1, j, N))
    return H

def noisy_func(noise_amplitude, perturb_times):
    random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
    func1 = lambda t: 0.5j*np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
    noisy_func1 = lambda t: func1(t + random_phase)
    return noisy_func1(perturb_times)


N = 2

omega = 2. * np.pi * 20

Omega_R = 2. * np.pi * 6

J = 1

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

result_t1 = mesolve(H0(omega, J, N), productstateZ(0, N-1, N), t1, [], Exps, options=opts)

result_t1t2 = mesolve(H0(omega, J, N), result_t1.states[timesteps - 1], t2, [], Exps, options=opts)

Perturb = MagnetizationX(N)
Measure = MagnetizationY(N)

result_AB = mesolve(H0(omega, J, N), Perturb * result_t1.states[timesteps - 1], t2, [], Exps, options=opts)

for t in range(0, timesteps):

    prod_AB = result_t1t2.states[t - 1].dag() * Measure * result_AB.states[t - 1]

    prod_BA = result_AB.states[t - 1].dag() * Measure * result_t1t2.states[t - 1]

    Commutator = prod_AB - prod_BA

    AntiCommutator = prod_AB + prod_BA

    Commutatorlist.append(1j * Commutator[0][0][0])
    Anticommutatorlist.append(AntiCommutator[0][0][0])
    #print('Commutator:', 1j * Commutator[0][0])
    #print('AntiCommutator: ', AntiCommutator[0][0])


freq = np.fft.fftfreq(t2.shape[-1])
plt.plot(freq, np.real(np.fft.fft(Commutatorlist)), linestyle='--', marker='o', markersize='5', label="Commutator")
plt.plot(freq, np.real(np.fft.fft(Anticommutatorlist)), linestyle='--', marker='o', markersize='5', label="Anticommutator")
plt.xlim(-1/2, 1/2)
plt.legend()
plt.show()

plt.plot(t2, np.real(Commutatorlist), label="Commutator")
plt.plot(t2, np.real(Anticommutatorlist),  label="Anticommutator")
plt.legend()
plt.xlabel('t_measure - t_perturb')
plt.show()

gamma1 = 0

def ohmic_spectrum(w):
    if w == 0.0:  # dephasing inducing noise
        return gamma1
    else:  # relaxation inducing noise
        return gamma1  # / 2 * (w / (2 * np.pi)) * (w > 0.0)

spectra_cb = [ohmic_spectrum]
a_ops = []


result_br = brmesolve(H0(omega, J, N), result_t1.states[timesteps-1], perturb_times, a_ops, spectra_cb=spectra_cb, options=opts)

result_t1t2_br = mesolve(H0(omega, J, N), result_br.states[timesteps - 1], t2, [], Exps, options=opts)


result_me = mesolve([H0(omega, J, N), [H1(Omega_R, N), S1], [H2(Omega_R, N), S1]], result_t1.states[timesteps - 1],
                    perturb_times, [0.6*sigmap(1, 0, N), 0.6*sigmam(1, 0, N)], Exps, options=opts)

result_t1t2_me = mesolve(H0(omega, J, N), result_me.states[timesteps - 1], t2, [], Exps, options=opts)


anan = np.zeros(timesteps)
upup = np.zeros(timesteps)
downdown = np.zeros(timesteps)
updown = np.zeros(timesteps)
downup = np.zeros(timesteps)
upupt1t2me = np.zeros(timesteps)
downdownt1t2me = np.zeros(timesteps)
updownt1t2me = np.zeros(timesteps)
downupt1t2me = np.zeros(timesteps)
upupbr = np.zeros(timesteps)
downdownbr = np.zeros(timesteps)
updownbr = np.zeros(timesteps)
downupbr = np.zeros(timesteps)
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
        upupt1t2me[t] = np.real(result_t1t2_me.states[t][0][0][0])
        downdownt1t2me[t] = np.real(result_t1t2_me.states[t][1][0][1])
        updownt1t2me[t] = np.real(result_t1t2_me.states[t][1][0][0])
        downupt1t2me[t] = np.real(result_t1t2_me.states[t][0][0][1])
        upupbr[t] = np.real(result_br.states[t][0][0][0])
        downdownbr[t] = np.real(result_br.states[t][1][0][1])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][0])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t][0][0][1])
        upupt1t2br[t] = np.real(result_t1t2_br.states[t][0][0][0])
        downdownt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][1])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t][1][0][0])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t][0][0][1])
else:
    for t in range(0, timesteps):
        anan[t] = np.real(result_me.states[t].ptrace(0)[0][0][0])
        upup[t] = np.real(result_me.states[t].ptrace(0)[1][0][1])
        downdown[t] = np.real(result_me.states[t].ptrace(0)[2][0][2])
        updown[t] = np.real(result_me.states[t].ptrace(0)[1][0][2])
        downup[t] = np.real(result_me.states[t].ptrace(0)[2][0][1])
        upupt1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[0][0][0])
        downdownt1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[1][0][1])
        updownt1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[1][0][0])
        downupt1t2me[t] = np.real(result_t1t2_me.states[t].ptrace(0)[0][0][1])
        upupbr[t] = np.real(result_br.states[t].ptrace(0)[0][0][0])
        downdownbr[t] = np.real(result_br.states[t].ptrace(0)[1][0][1])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][0])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[0][0][1])
        upupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[0][0][0])
        downdownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][1])
        updownt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[1][0][0])
        downupt1t2br[t] = np.real(result_t1t2_br.states[t].ptrace(0)[0][0][1])


for noise_amplitude in np.logspace(-2, -2, num=1):

    i = 1
    random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
    S = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(noise_amplitude, perturb_times))

    #print('H0...')
    #print(H0(omega, J, N))
    #print('H1...')
    #print(H1(Omega_R, N))
    #print('H2...')
    #print(H2(Omega_R, N))

    result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], result_t1.states[timesteps - 1],
                      perturb_times, e_ops=Exps, options=opts)

    #opts = Options(store_states=True, store_final_state=True, rhs_reuse=True)
    states2 = np.array(result2.states[timesteps - 1])
    expect2 = np.array(result2.expect[:])
    while i < 20:
        print(i)
        i += 1
        random_phase = noise_amplitude * np.random.randn(perturb_times.shape[0])
        S = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_func(noise_amplitude, perturb_times))

        result2 = mesolve([H0(omega, J, N), [H1(Omega_R, N), S], [H2(Omega_R, N), S]], result_t1.states[timesteps - 1],
                          perturb_times, e_ops=Exps, options=opts)

        states2 += np.array(result2.states[timesteps - 1])
        expect2 += np.array(result2.expect[:])

    func2 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
    noisy_func2 = lambda t: func2(t + random_phase)
    noisy_data2 = noisy_func2(perturb_times)
    S2 = Cubic_Spline(perturb_times[0], perturb_times[-1], noisy_data2)

    states2 = states2/i
    expect2 = expect2/i
    #print(Qobj(states2))
    #print((expect2[5]+expect2[8]).mean())
    density_matrix = Qobj([[expect2[5][timesteps - 1], expect2[6][timesteps - 1]], [expect2[7][timesteps - 1], expect2[8][timesteps - 1]]])
    #print(density_matrix)
    result3 = mesolve(H0(omega, J, N), Qobj(states2), t2, [], e_ops=Exps, options=opts)

    #print('Initial state ....')
    #print(productstateZ(0, 0, N))
    #print(productstateZ(0, 0, N).dag()*sigmaz(1, N)*productstateZ(0, 0, N))




    #print('Commutator:', 1j * Commutator[0][0])
    #print('AntiCommutator: ', AntiCommutator[0][0])

    fig, ax = plt.subplots(5, 2, figsize=(10, 10))
    ax[0, 0].plot(perturb_times, func2(perturb_times))
    ax[0, 0].plot(perturb_times, noisy_data2, 'o')
    ax[0, 0].plot(perturb_times, S2(perturb_times), lw=2)
    ax[0, 0].set_xlabel('Time [1/J]')
    ax[0, 0].set_ylabel('Coupling Amplitude')
    ax[0, 0].set_xlim([0, 0.1])

    ax[0, 1].plot(perturb_times, S2(perturb_times), lw=2)
    ax[0, 1].set_xlabel('Time [1/J]')


    ax[1, 0].plot(t1, np.real(result_t1.expect[1]), label="MagnetizationZ")
    ax[1, 0].plot(t1, np.real(result_t1.expect[2]), label="MagnetizationY")
    ax[1, 0].plot(t1, np.real(result_t1.expect[0]), label="MagnetizationX")
    #ax[1, 0].plot(t1, result_t1.expect[3], label="tensor(SigmaZ,Id) ")
    #ax[1, 0].plot(t1, result_t1.expect[4], label="tensor(Id,SigmaZ) ")
    ax[1, 0].set_xlabel('Free Evolution Time [1/J]')
    ax[1, 0].set_ylabel('Magnetization')
    #ax[1, 0].legend(loc="upper right")
    ax[1, 0].set_ylim([-1.1, 1.1])


    ax[1, 1].plot(t2, np.real(result_AB.expect[1]), label="MagnetizationZ")
    ax[1, 1].plot(t2, np.real(result_AB.expect[2]), label="MagnetizationY")
    ax[1, 1].plot(t2, np.real(result_AB.expect[0]), label="MagnetizationX")
    #ax[1, 1].plot(t2, result_AB.expect[3], label="tensor(SigmaZ,Id)")
    #ax[1, 1].plot(t2, result_AB.expect[4], label="tensor(Id,SigmaZ)")
    ax[1, 1].set_xlabel('After Perturbation Operator [1/J]')
    ax[1, 1].legend(loc="right")
    ax[1, 1].set_ylim([-1.1, 1.1])

    #ax[2, 0].plot(perturb_times, expect2[0], label="MagnetizationX")
    ax[2, 0].plot(perturb_times, expect2[1], label="MagnetizationZ")
    #ax[2, 0].plot(perturb_times, expect2[2], label="MagnetizationY")
    #ax[2, 0].plot(perturb_times, expect2[3], label="tensor(SigmaZ,Id) ")
    #ax[2, 0].plot(perturb_times, expect2[4], label="tensor(Id,SigmaZ) ")
    ax[2, 0].plot(perturb_times, expect2[5], label="upup")
    ax[2, 0].plot(perturb_times, expect2[6], label="updown")
    ax[2, 0].plot(perturb_times, expect2[7], label="downup")
    ax[2, 0].plot(perturb_times, expect2[8], label="downdown")
    ax[2, 0].plot(perturb_times, expect2[9], label="aa")
    ax[2, 0].set_xlabel('Time Dependent Perturbation [1/J]')
    #ax[2, 0].legend(loc="right")
    ax[2, 0].set_ylim([-1.1, 1.1])

    #ax[2, 1].plot(t2, result3.expect[0], label="MagnetizationX")
    ax[2, 1].plot(t2, np.real(result3.expect[1]), label="MagnetizationZ")
    #ax[2, 1].plot(t2, result3.expect[2], label="MagnetizationY")
    #ax[2, 1].plot(t2, result3.expect[3], label="tensor(SigmaZ,Id) ")
    #ax[2, 1].plot(t2, result3.expect[4], label="tensor(Id,SigmaZ) ")
    ax[2, 1].plot(t2, np.real(result3.expect[5]), label="upup")
    ax[2, 1].plot(t2, np.real(result3.expect[6]), label="updown")
    ax[2, 1].plot(t2, np.real(result3.expect[7]), label="downup")
    ax[2, 1].plot(t2, np.real(result3.expect[8]), label="downdown")
    ax[2, 1].plot(t2, np.real(result3.expect[9]), label="aa")
    ax[2, 1].set_xlabel('After Time Dependent Pertubation [1/J]')
    ax[2, 1].legend(loc="right")
    ax[2, 1].set_ylim([-1.1, 1.1])

    #ax[3, 0].plot(t2, result_AB_me.expect[0], label="MagnetizationX")
    ax[3, 0].plot(t2, np.real(result_me.expect[1]), label="MagnetizationZ")
    #ax[3, 0].plot(t2, result_AB_me.expect[2], label="MagnetizationY")
    ax[3, 0].plot(t2, upup, label="uu")
    ax[3, 0].plot(t2, updown, label="ud")
    ax[3, 0].plot(t2, downup, label="du")
    ax[3, 0].plot(t2, downdown, label="dd")
    ax[3, 0].plot(t2, anan, label="aa")
    #ax[3, 0].plot(t2, result_AB.expect[3], label="tensor(SigmaZ,Id)")
    #ax[3, 0].plot(t2, result_AB.expect[4], label="tensor(Id,SigmaZ)")
    ax[3, 0].set_xlabel('Lindblad Perturbation [1/J]')
    #ax[3, 0].legend(loc="right")
    ax[3, 0].set_ylim([-1.1, 1.1])

    #ax[3, 1].plot(t2, result_t1t2_me.expect[0], label="MagnetizationX")
    ax[3, 1].plot(t2, np.real(result_t1t2_me.expect[1]), label="MagnetizationZ")
    #ax[3, 1].plot(t2, result_t1t2_me.expect[2], label="MagnetizationY")
    ax[3, 1].plot(t2, upupt1t2me, label="upup")
    ax[3, 1].plot(t2, updownt1t2me, label="updown")
    ax[3, 1].plot(t2, downupt1t2me, label="downup")
    ax[3, 1].plot(t2, downdownt1t2me, label="downdown")
    #ax[3, 1].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
    #ax[3, 1].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
    ax[3, 1].set_xlabel('After Lindblad Perturbation [1/J]')
    ax[3, 1].legend(loc="right")
    ax[3, 1].set_ylim([-1.1, 1.1])

    #ax[4, 0].plot(t2, result_t1t2_br.expect[0], label="MagnetizationX")
    ax[4, 0].plot(t2, np.real(result_t1t2_br.expect[1]), label="MagnetizationZ")
    #ax[4, 0].plot(t2, result_t1t2_br.expect[2], label="MagnetizationY")
    ax[4, 0].plot(t2, upupbr, label="upup")
    ax[4, 0].plot(t2, updownbr, label="updown")
    ax[4, 0].plot(t2, downupbr, label="downup")
    ax[4, 0].plot(t2, downdownbr, label="downdown")
    #ax[4, 0].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
    #ax[4, 0].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
    ax[4, 0].set_xlabel('Bloch Redfield Perturbation [1/J]')
    #ax[4, 0].legend(loc="right")
    ax[4, 0].set_ylim([-1.1, 1.1])

    #ax[4, 1].plot(t2, result_t1t2_br.expect[0], label="MagnetizationX")
    ax[4, 1].plot(t2, np.real(result_t1t2_br.expect[1]), label="MagnetizationZ")
    #ax[4, 1].plot(t2, result_t1t2_br.expect[2], label="MagnetizationY")
    ax[4, 1].plot(t2, upupt1t2br, label="upup")
    ax[4, 1].plot(t2, updownt1t2br, label="updown")
    ax[4, 1].plot(t2, downupt1t2br, label="downup")
    ax[4, 1].plot(t2, downdownt1t2br, label="downdown")
    #ax[4, 1].plot(t2, result_t1t2.expect[3], label="tensor(SigmaZ,Id)")
    #ax[4, 1].plot(t2, result_t1t2.expect[4], label="tensor(Id,SigmaZ)")
    ax[4, 1].set_xlabel('After Bloch Redfield Pertubation [1/J]')
    ax[4, 1].legend(loc="right")
    ax[4, 1].set_ylim([-1.1, 1.1])
    fig.tight_layout()
    plt.show()
    #plt.savefig("Dephasing with"+str(noise_amplitude)+".pdf")
