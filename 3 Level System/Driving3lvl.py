import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter


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
    func2 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
    return func2(perturb_times)


def noisy_func(noise_amplitude, perturb_times, omega, bandwidth):
    ########  STARTING FUNCTION  #####################
    func1 = lambda t: 0.5j * np.exp(-1j * t * 1 * omega) - 0.5j * np.exp(1j * t * 1 * omega)
    fourier = np.abs(np.fft.fft(func1(perturb_times)))

    max_neg = len(perturb_times) - np.argmax(fourier[0: int(len(perturb_times) / 2)])
    max_pos = int(len(perturb_times) / 2) - np.argmax(fourier[int(len(perturb_times) / 2): len(perturb_times)])

    ########### AMPLITUDE NOISE ################
    random_amplitude = np.random.normal(0, noise_amplitude, size=len(perturb_times))

    #######  FILTERING  ####################
    # random_amplitude = butter_bandpass_filter(random_amplitude, 25, 45, len(random_amplitude)/perturb_times[-1], order=6)
    '''
    noisefreq = np.fft.fft(random_amplitude)
    #noisefreq = np.max(fourier)*np.ones_like(perturb_times)/100#

    noisefreq[max_neg + bandwidth: max_neg - bandwidth] =\
        envelope("Blackman", noisefreq[max_neg + bandwidth: max_neg - bandwidth])

    noisefreq[max_pos + bandwidth: max_pos - bandwidth] =\
        envelope("Blackman", noisefreq[max_pos + bandwidth: max_pos - bandwidth])

    noisefreq[0: max_pos - bandwidth] = 0
    noisefreq[max_pos + bandwidth: max_neg - bandwidth] = 0
    noisefreq[max_neg + bandwidth: len(perturb_times)] = 0

    #noisefreq[max_neg - bandwidth] = 1000
    #noisefreq[max_neg + bandwidth] = 1000
    #noisefreq[max_pos + bandwidth] = 1000
    #noisefreq[max_pos - bandwidth] = 1000
    random_amplitude = np.fft.ifft(noisefreq)
    '''

    ######### FREQUENCY NOISE #####################
    # random_frequency = np.random.uniform(low=0.8, high=1.2, size=perturb_times.shape[0])

    ########### PHASE NOISE ##############
    random_phase = np.zeros_like(perturb_times)
    i = 0
    time = np.random.randint(0, len(perturb_times) - 1)
    times = [time]
    random_phase[time] = noise_amplitude * np.random.uniform(-np.pi, np.pi)

    number_of_jumps = []
    while i < 2:
        i += 1
        time = np.random.randint(0, len(perturb_times) - 1)
        # print(times)
        if np.min(np.abs(times - np.full_like(times, time))) > len(perturb_times) / 100:
            random_phase[time] = noise_amplitude * np.random.uniform(-np.pi, np.pi)
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
