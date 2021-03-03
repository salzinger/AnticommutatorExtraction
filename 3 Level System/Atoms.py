from qutip import *
import numpy as np


def threebasis():
    return np.array([basis(3, 0), basis(3, 1), basis(3, 2)], dtype=object)


def productstateZ(up_atom, down_atom, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [down for _ in oplist]
    oplist[up_atom] = Qobj(up)
    oplist[down_atom] = Qobj(down)
    return tensor(oplist)


def productstateA(up_atom, ancilla_atom, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [Qobj(down) for _ in oplist]
    oplist[up_atom] = Qobj(up)
    oplist[ancilla_atom] = Qobj(ancilla)
    return tensor(Qobj(up), Qobj(ancilla)) + tensor(Qobj(ancilla), Qobj(up)) + tensor(Qobj(down), Qobj(ancilla)) + \
           tensor(Qobj(ancilla), Qobj(down)) + tensor(Qobj(ancilla), Qobj(ancilla))


def productstateX(m, j, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [Qobj((up + down)).unit() for _ in oplist]
    return tensor(oplist)


def anan(m, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    oplist[m] = Qobj(ancilla * ancilla.conj().T)
    return tensor(oplist)


def upup(m, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    oplist[m] = Qobj(up * up.conj().T)
    return tensor(oplist)


def downdown(m, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    oplist[m] = Qobj(down * down.conj().T)
    return tensor(oplist)


def sigmap(ancilla_coupling, m, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    if ancilla_coupling:
        oplist[m] = Qobj(ancilla * up.conj().T)
    else:
        oplist[m] = Qobj(up * down.conj().T)
    return tensor(oplist)


def sigmam(ancilla_coupling, m, N):
    ancilla, up, down = threebasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(3) for _ in oplist]
    if ancilla_coupling:
        oplist[m] = Qobj(up * ancilla.conj().T)
    else:
        oplist[m] = Qobj(down * up.conj().T)
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
    return -sum / N


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
