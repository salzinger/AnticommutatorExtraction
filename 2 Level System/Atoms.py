from qutip import *
import numpy as np


def twobasis():
    return np.array([basis(2, 0), basis(2, 1)], dtype=object)


def productstateZ(up_atom, down_atom, N):
    up, down = twobasis()
    oplist = np.empty(N, dtype=object)
    oplist = [Qobj(down) for _ in oplist]
    oplist[up_atom] = Qobj(up)
    oplist[down_atom] = Qobj(down)
    return tensor(oplist)


def productstateX(m, j, N):
    ancilla, up, down = twobasis()
    oplist = np.empty(N, dtype=object)
    oplist = [Qobj((up + down)).unit() for _ in oplist]
    return tensor(oplist)


def bellstate(i, j, N):
    blist1 = []
    blist2 = []
    for n in range(0, N - 2):
        blist1.append(identity(2))
        blist2.append(identity(2))

    blist1.insert(i, basis(2, 0))
    blist1.insert(j, basis(2, 1))

    blist2.insert(i, basis(2, 1))
    blist2.insert(j, basis(2, 0))

    bell = tensor(blist1) + tensor(blist2)

    return bell.unit()


def upup(m, N):
    up, down = twobasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(2) for _ in oplist]
    oplist[m] = Qobj(up) * Qobj(up).dag()
    return tensor(oplist)


def downdown(m, N):
    up, down = twobasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(2) for _ in oplist]
    oplist[m] = Qobj(down) * Qobj(down).dag()
    return tensor(oplist)


def sigmap(m, N):
    up, down = twobasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(2) for _ in oplist]
    oplist[m] = Qobj(up) * Qobj(down).dag()
    return tensor(oplist)


def sigmam(m, N):
    up, down = twobasis()
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(2) for _ in oplist]
    oplist[m] = Qobj(down) * Qobj(up).dag()
    return tensor(oplist)


def sigmaz(j, N):
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(2) for _ in oplist]
    oplist[j] = Qobj([[1, 0], [0, -1]])
    return tensor(oplist)


def sigmax(j, N):
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(2) for _ in oplist]
    oplist[j] = Qobj([[0, 1], [1, 0]])
    return tensor(oplist)


def sigmay(j, N):
    oplist = np.empty(N, dtype=object)
    oplist = [qeye(2) for _ in oplist]
    oplist[j] = Qobj([[0, -1j], [1j, 0]])

    return tensor(oplist)


def MagnetizationZ(N):
    sum = 0
    for j in range(0, N):
        sum += sigmaz(j, N)
    return sum / N


def MagnetizationX(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(j, N)
    return sum / N


def MagnetizationY(N):
    sum = 0
    for j in range(0, N):
        sum += sigmay(j, N)
    return sum / N


def H0(omega, J, N):
    H = 0
    for j in range(0, N):
        H += 1 * omega / 2 * sigmaz(j, N)
        for i in range(0, N):
            if i != j:
                H += J * (sigmax(i, N) * sigmax(j, N) + sigmay(i, N) * sigmay(j, N))
    return H


def H1(Omega_R, N):
    H = 0
    for j in range(0, N):
        H -= Omega_R * (sigmap(j, N))
    return H


def H2(Omega_R, N):
    H = 0
    for j in range(0, N):
        H -= Omega_R * (sigmam(j, N))
    return H
