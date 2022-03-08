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
    up, down = twobasis()
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
    return sum / N / 2


def MagnetizationX(N):
    sum = 0
    for j in range(0, N):
        sum += sigmax(j, N)
    return sum / N / 2


def MagnetizationY(N):
    sum = 0
    for j in range(0, N):
        sum += sigmay(j, N)
    return sum / N / 2


def H0(omega, J, N):
    H = 0
    for j in range(0, N):
        H += 1 * omega / 2 * sigmaz(j, N) + Qobj([[0, 0], [0, 2*np.pi*(-0.07)]])
        for i in range(0, N):
            if i != j:
                H += J * (sigmap(i, N) * sigmam(j, N) + sigmam(i, N) * sigmap(j, N)) / (np.random.uniform(0.65, 1.35)*np.abs(i-j))**3
    return H


def H1(Omega_R, N):
    H = 0
    for j in range(0, N):
        H -= Omega_R * (sigmap(j, N))
    #print(H)
    return H


def H2(Omega_R, N):
    H = 0
    for j in range(0, N):
        H -= Omega_R * (sigmam(j, N))
    return H
