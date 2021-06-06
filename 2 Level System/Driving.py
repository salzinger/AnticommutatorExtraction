from scipy import signal
from scipy.signal import butter, lfilter
from pylab import plot, show, grid, xlabel, ylabel
from math import sqrt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def lorentzian(frequencies, amplitude, omega_0, gamma):
    func = lambda omega: amplitude/gamma/np.pi/(2*((omega-omega_0)/gamma)**2 + 1/2)
    return func(frequencies)


def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta * sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def envelope(shape, function):
    if shape == "Blackman":
        window = signal.windows.blackman(len(function))
        function = window * function
    elif shape == "Window":
        window = np.ones_like(function)
        function = window * function
    else:
        None
    return function


def func(perturb_times, omega):
    func2 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega)# - 0.5j * np.exp(1j * t * 1 * omega)
    return func2(perturb_times)


def noisy_func(gamma, perturb_times, omega, bath):
    if bath == 'Forward3MHzcsv.txt':
        data = np.loadtxt('Forward3MHzcsv.txt')
        data_reversed = -data[::-1]

        data = np.cumsum(data)
        data_reversed = np.cumsum(data_reversed)+data[-1]+180

        data = np.append(data, data_reversed)
        #plt.plot(np.linspace(0, 0.2, int(len(data))), data, color="red")
        # plt.plot(np.linspace(0.1, 0.2, int(len(data))), np.cumsum(-data_reversed)+np.cumsum(data)[-1])
        #plt.ylabel('Phase [°]')
        #plt.xlabel('Time [us]')
        #plt.legend()
        #plt.show()
        #print(data)
        #print(perturb_times)
        #func1 = lambda t: 0.5j * np.exp(-1j * t * omega) - 0.5j * np.exp(1j * t * omega)
        func1 = lambda t: np.exp(-1j * t * omega)/2
        return func1(perturb_times*1+data/omega*2*np.pi/360)

    elif bath == "markovian":
        # Total time.
        T = perturb_times[-1]
        # Number of steps.
        N = len(perturb_times)
        # Time step size
        dt = T / N
        # Number of realizations to generate.
        m = 1
        # Create an empty array to store the realizations.
        x = np.empty((m, N + 1))
        # Initial values of x.
        x[:, 0] = 0

        phase_noise = brownian(x[:, 0], N, dt, np.sqrt(gamma), out=x[:, 1:])

        #t = np.linspace(0.0, N * dt, N)
        #for k in range(m):
        #    plot(t, phase_noise[k])
        #plot(t, delta*t)
        #plot(t, -delta*t)
        #xlabel('t', fontsize=16)
        #ylabel('phase', fontsize=16)
        #grid(True)
        #show()


        #print(perturb_times)
        #print(phase_noise[0])
        #print(perturb_times+phase_noise)
        #lab_frame:
        func1 = lambda t: 0.5j * np.exp(-1j * t * omega) - 0.5j * np.exp(1j * t * omega)
        return func1(perturb_times+phase_noise[0]/omega*(np.pi/2)**2)
        #rotating_frame:
        #func1 = lambda t: np.exp(-1j * t * omega)
        #return func1(phase_noise[0] / omega)
    elif bath == "scale-free":
        func_array = np.zeros_like(perturb_times) + 1j * np.zeros_like(perturb_times)
        for d_omega in np.linspace(0, gamma, 100):
            func = lambda t: 0.5j * np.exp(-1j * t * (omega - d_omega)) - 0.5j * np.exp(1j * t * (omega - d_omega))
            func_array += func(perturb_times)
        return func_array
    elif bath == "random":
        ########  STARTING FUNCTION  #####################
        func1 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
        fourier = np.abs(np.fft.fft(func1(perturb_times)))

        bandwidth = int(gamma)

        max_neg = len(perturb_times) - np.argmax(fourier[0: int(len(perturb_times) / 2)])
        max_pos = int(len(perturb_times) / 2) - np.argmax(fourier[int(len(perturb_times) / 2): len(perturb_times)])

        ########### AMPLITUDE NOISE ################
        random_amplitude = np.random.normal(0, gamma, size=len(perturb_times))

        #######  FILTERING  ####################
        # random_amplitude = butter_bandpass_filter(random_amplitude, 25, 45, len(random_amplitude)/perturb_times[-1], order=6)

        noisefreq = np.fft.fft(random_amplitude)
        # noisefreq = np.max(fourier)*np.ones_like(perturb_times)/100#

        noisefreq[max_neg + bandwidth: max_neg - bandwidth] = \
            envelope("Window", noisefreq[max_neg + bandwidth: max_neg - bandwidth])

        noisefreq[max_pos + bandwidth: max_pos - bandwidth] = \
            envelope("Window", noisefreq[max_pos + bandwidth: max_pos - bandwidth])

        noisefreq[0: max_pos - bandwidth] = 0
        noisefreq[max_pos + bandwidth: max_neg - bandwidth] = 0
        noisefreq[max_neg + bandwidth: len(perturb_times)] = 0

        # noisefreq[max_neg - bandwidth] = 1000
        # noisefreq[max_neg + bandwidth] = 1000
        # noisefreq[max_pos + bandwidth] = 1000
        # noisefreq[max_pos - bandwidth] = 1000
        random_amplitude = np.fft.ifft(noisefreq)

        ######### FREQUENCY NOISE #####################
        # random_frequency = np.random.uniform(low=0.8, high=1.2, size=perturb_times.shape[0])

        ########### PHASE NOISE ##############
        random_phase = np.zeros_like(perturb_times)
        i = 0
        time = np.random.randint(0, len(perturb_times) - 1)
        times = [time]
        random_phase[time] = gamma * np.random.uniform(-np.pi, np.pi)

        number_of_jumps = []
        while i < 2:
            i += 1
            time = np.random.randint(0, len(perturb_times) - 1)
            # print(times)
            if np.min(np.abs(times - np.full_like(times, time))) > len(perturb_times) / 100:
                random_phase[time] = gamma * np.random.uniform(-np.pi, np.pi)
                # random_amplitude[time] = noise_amplitude * np.random.uniform(-1, 1)
                times.append(time)
        # random_phase = noise_amplitude * np.random.uniform(low=- np.pi, high=np.pi, size=perturb_times.shape[0])# + np.pi
        # print(len(times))
        for t in range(0, len(random_phase) - 1):
            if random_phase[t] == 0:  # divmod(t, np.random.randint(200, 300))[1] != 0:
                # random_amplitude[t+1] = random_amplitude[t]
                random_phase[t + 1] = random_phase[t]
                # random_phase[t] = random_phase[t-1]
                # random_frequency[t + 1] = random_frequency[t]

        noisy_func1 = lambda t: func1(t) + random_amplitude
        return noisy_func1(perturb_times)


def ohmic_spectrum(w, gamma1):
    if w == 0.0:  # dephasing inducing noise
        return gamma1
    else:  # relaxation inducing noise
        return gamma1  # / 2 * (w / (2 * np.pi)) * (w > 0.0)

    # The Wiener process parameter.
#delta = 2
    # Total time.
#T = 10.0
    # Number of steps.
#N = 500
    # Time step size
#dt = T / N
    # Number of realizations to generate.
#m = 1
    # Create an empty array to store the realizations.
#x = np.empty((m, N + 1))
    # Initial values of x.
#x[:, 0] = 0

#brownian(x[:, 0], N, dt, delta, out=x[:, 1:])

#t = np.linspace(0.0, N * dt, N + 1)
#for k in range(m):
#    plot(t, x[k])
#xlabel('t', fontsize=16)
#ylabel('x', fontsize=16)
#grid(True)
#show()