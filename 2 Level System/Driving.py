from scipy import signal
from scipy.signal import butter, lfilter
from pylab import plot, show, grid, xlabel, ylabel
from math import sqrt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def get_fft(data, x=None):
    if x is None:
        x = np.arange(len(data))
    # build frequency array
    n_samples = len(data)
    fs = (x[1] - x[0])
    f = fftfreq(n_samples, fs)

    # compute fft
    ff = fft(data) / n_samples * 2
    return f, ff

def average_psd(gamma,omega,samples,sample_time,averages):
    long = sqrt(2) * noisy_func(gamma, np.linspace(0, sample_time, samples), omega, "markovian")
    fs = samples / sample_time
    F, P = signal.welch(
        long, fs,
        nperseg=samples)
    i=1
    for x in range(0, averages-1):
        i+=1
        print(i)
        #    long = np.append(long, noisy_func(gamma, perturb_times, omega, bath)[0:int(len(perturb_times) / 2 - 10)])
        #    long_nn = np.append(long, func(perturb_times, omega)[0:int(len(perturb_times) / 2 - 10)])
        long = sqrt(2) * noisy_func(gamma, np.linspace(0, sample_time, samples), omega, "markovian")

        f, Pxx_den = signal.welch(
            long, fs,
            nperseg=samples)

        F += f
        P += Pxx_den

    return F / averages, P / averages

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


def davies_harte(T, N, H, y):
    '''
    Generates sample paths of fractional Brownian Motion using the Davies Harte method

    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k, H: 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    g = [gamma(k, H) for k in range(0, N)];
    r = g + [0] + g[::-1][0:N - 1]

    # Step 1 (eigenvalues)
    j = np.arange(0, 2 * N);
    k = 2 * N - 1
    lk = np.fft.fft(r * np.exp(2 * np.pi * complex(0, 1) * k * j * (1 / (2 * N))))[::-1]

    # Step 2 (get random variables)
    Vj = np.zeros((2 * N, 2), dtype=np.complex);
    Vj[0, 0] = np.random.standard_normal();
    Vj[N, 0] = np.random.standard_normal()

    for i in range(1, N):
        Vj1 = np.random.standard_normal();
        Vj2 = np.random.standard_normal()
        Vj[i][0] = Vj1;
        Vj[i][1] = Vj2;
        Vj[2 * N - i][0] = Vj1;
        Vj[2 * N - i][1] = Vj2

    # Step 3 (compute Z)
    wk = np.zeros(2 * N, dtype=np.complex)
    wk[0] = np.sqrt((lk[0] / (2 * N))) * Vj[0][0];
    wk[1:N] = np.sqrt(lk[1:N] / (4 * N)) * ((Vj[1:N].T[0]) + (complex(0, 1) * Vj[1:N].T[1]))
    wk[N] = np.sqrt((lk[0] / (2 * N))) * Vj[N][0]
    wk[N + 1:2 * N] = np.sqrt(lk[N + 1:2 * N] / (4 * N)) * (
                np.flip(Vj[1:N].T[0]) - (complex(0, 1) * np.flip(Vj[1:N].T[1])))

    Z = np.fft.fft(wk);
    fGn = Z[0:N]
    fBm = np.cumsum(fGn) * (N ** (-H))
    fBm = (T ** H) * (fBm)
    path = np.array([0] + list(fBm))
    print(len(path[0:N]))
    return path[0:N]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
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
    if omega == 0:
        return np.ones_like(perturb_times)/2
    else:
        func2 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
        return func2(perturb_times)



def noisy_func(gamma, perturb_times, omega, bath, rise_time=0, second_rise_time=0):
    if bath == 'Forward3MHzcsv.txt':

        #there
        data = np.loadtxt('Forward3MHzcsv.txt')
        data_reversed = -data[::-1]
        print("noisy func fs: ", len(perturb_times)/perturb_times[-1])
        data = np.cumsum(data)

        ###and back again
        data_reversed = np.cumsum(data_reversed)+data[-1]+180

        data = np.append(data, data_reversed)

        #print(len(data))
        #plt.plot(np.linspace(0, 0.2, int(len(data))), data/180, color="#85bb65", linewidth="0.5")
        #plt.ylabel('Phase [$\pi$]', fontsize=16)
        #plt.xlabel('Time [us]', fontsize=16)
        #plt.legend()
        #plt.show()
        #print(data)
        #print(perturb_times)
        #func1 = lambda t: 0.5j * np.exp(-1j * t * omega) - 0.5j * np.exp(1j * t * omega)
        if omega == 0:
            return np.exp(-1j * data * 2 * np.pi/360)/2 #
            #return butter_bandpass_filter(np.exp(-1j * data * 2 * np.pi/360)/2, 0.01, 100, len(perturb_times)/perturb_times[-1], order=3)
        else:
            func1 = lambda t: np.exp(-1j * t * omega)/2
            #return func1(perturb_times+data/omega*2*np.pi/360)
            return butter_bandpass_filter(func1(perturb_times+data/omega*2*np.pi/360), 0.01, 31999, len(perturb_times)/perturb_times[-1], order=3)


    if bath == '10MHz_gamma.txt':


        #data = np.loadtxt('10MHz_gamma.txt')
        #data_reversed = -data[::-1]
        #data = np.append(data, 180)
        #data = np.append(data, data_reversed)
        #lines = np.array(data)
        '''
        with open('10MHz_gamma_frontandback.txt', 'w') as f:
            for line in lines:
                f.write(str(line))
                f.write('\n')
        '''
        data = np.loadtxt('10MHz_gamma.txt')
        #print(len(data))
        data_reversed = -data[::-1]

        #there
        data = np.cumsum(data)
        #print(butter_bandpass_filter(np.exp(-1j * data * 2 * np.pi/360)/2, 0.01, len(data)/32-1, len(data)/6, order=3))
        signal = np.array(butter_bandpass_filter(np.exp(-1j * data * 2 * np.pi/360)/2, 0.01, len(data)/50-1, len(data), order=1))

        unfiltered_signal = np.array(np.exp(-1j * data * 2 * np.pi/360)/2)
        #print(len(signal))

        #and back again
        data_reversed = np.cumsum(data_reversed)+data[-1]-180
        signal_reversed = -unfiltered_signal[::-1]#*np.exp(-0.1j)


        '''
        signal_reversed[5:-1] = signal_reversed[1:len(signal_reversed)-5]
        signal_reversed[1:5] = signal_reversed[0]
        '''

        #print(np.array(butter_bandpass_filter(np.exp(-1j * data_reversed * 2 * np.pi/360)/2, 0.01, len(data)/2-1, len(data), order=3)))
        signal = np.append(unfiltered_signal, signal_reversed)
        #signal = np.append(signal, np.array(butter_bandpass_filter(np.exp(-1j * data_reversed * 2 * np.pi/360)/2, 0.01, len(data)/2-1, len(data), order=3)))
        #print(len(signal))
        data = np.append(data, data_reversed)



        rise_time1=2100


        signal[0:rise_time]=(1-np.exp(-np.linspace(0,5,rise_time)))*signal[0:rise_time]

        #rise_time=2250

        signal[int(len(signal)/2):int(len(signal)/2)+second_rise_time] = (1 - np.exp(-np.linspace(0, 5, second_rise_time))) * signal[int(len(signal)/2):int(len(signal)/2)+second_rise_time]

        #print(len(data))


        #plt.plot(np.linspace(0, 0.2, int(len(data))), data/180, color="#85bb65", linewidth="0.5")
        #plt.ylabel('Phase [$\pi$]', fontsize=16)
        #plt.xlabel('Time [us]', fontsize=16)
        #plt.legend()
        #plt.show()
        #print(data)
        #print(perturb_times)
        #func1 = lambda t: 0.5j * np.exp(-1j * t * omega) - 0.5j * np.exp(1j * t * omega)
        if omega == 0:


            #func1 = lambda t: (1-np.exp(-100*t+0.001))*np.exp(-1j * t)/2
            #return func1(perturb_times+data/omega*2*np.pi/360)
            #return func1(perturb_times+data*2*np.pi/360)
            if rise_time==0:
                return  np.exp(-1j * data * 2 * np.pi / 360) / 2
            else:
                return butter_bandpass_filter(np.exp(-1j * data * 2 * np.pi/360)/2, 0.01, rise_time*len(data)/64, len(data), order=3)
            #return butter_bandpass_filter(signal, 0.01, len(data)/550-1, len(data), order=2)
                #return signal
            #return np.exp(-1j * data * 2 * np.pi/360)/2 * np.exp(-np.linspace(0,6,len(data)) * 0.15)
            #func1(perturb_times + data * 2 * np.pi / 360)
        else:
            func1 = lambda t: np.exp(-1j * t * omega)/2
            #return func1(perturb_times+data/omega*2*np.pi/360)
            return butter_bandpass_filter(func1(perturb_times+data/omega*2*np.pi/360), 0.01, 31999, len(perturb_times)/perturb_times[-1], order=3)

    elif bath == "fbm":
        # Total time.
        T = perturb_times[-1]
        # Number of steps.
        N = len(perturb_times)
        # Time step size
        dt = T / N

        phase_noise = davies_harte(perturb_times[-1], len(perturb_times), 1/2, gamma)
        '''
        Generates sample paths of fractional Brownian Motion using the Davies Harte method

        args:
            T:      length of time (in years)
            N:      number of time steps within timeframe
            H:      Hurst parameter
        '''
        func1 = lambda t: np.exp(-1j * t * omega)/2

        if omega == 0:
            return np.exp(-1j * phase_noise[0])/2
        else:
            return func1(perturb_times+phase_noise[0]/omega)/2  #*(np.pi/2)**2

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
        #func1 = lambda t: 0.5j * np.exp(-1j * t * omega) - 0.5j * np.exp(1j * t * omega)
        if omega == 0:
            return np.exp(-1j * phase_noise[0])/2
        else:
            func1 = lambda t: np.exp(-1j * t * omega)
            return func1(perturb_times+phase_noise[0]/omega*(np.pi/2)**2)/2
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