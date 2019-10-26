import scipy.fftpack
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.fftpack
import scipy.signal
import wave
from matplotlib.backends.backend_pdf import PdfPages

# Read mat file
fname = 'speechToken.mat'
# Fetch context in mat
mat = scipy.io.loadmat(fname)
pprint(mat)
# We want tokenS
print(mat['tokenS'].shape)

# Get sounds
sounds = [e[0] for e in mat['tokenS']]
print(sounds)
num = len(sounds)
length = len(sounds[0][0])
print(num, length)

# Plot waveforms

framerate = 44100
fs = 1 / framerate
freq_range = (300, 3000)
print(length / framerate)
times = np.array(range(length)) / framerate


def myfft(data, fs=fs):
    fft = scipy.fftpack.fft(data)
    psd = np.abs(fft) ** 2
    fftfreq = scipy.fftpack.fftfreq(len(data), fs)
    return fft, psd, fftfreq


def myifft(fft, fftfreq):
    fft = fft.copy()
    fft[np.abs(fftfreq) < freq_range[0]] = 0
    fft[np.abs(fftfreq) > freq_range[1]] = 0
    return np.real(scipy.fftpack.ifft(fft)), fft


def mywavelet(sig, times=times, framerate=framerate):
    widths = [framerate/e for e in range(300, 3000, 100)]
    cwtmatr = scipy.signal.cwt(sig, scipy.signal.ricker, widths)
    fig, axe = plt.subplots(1, 1)
    x = axe.imshow(cwtmatr,
                   extent=[min(times), max(times), min(widths), max(widths)],
                   cmap='PRGn', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    fig.colorbar(x, ax=axe)
    return cwtmatr, fig


figs = []
for j, s in enumerate(sounds):
    s = s.copy().ravel()
    fft, psd, fftfreq = myfft(s)
    select = fftfreq > 0

    fig, axes = plt.subplots(2, 2, figsize=(10, 4))
    ax = axes[0, 0]
    ax.plot(times, s, label=j)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(fftfreq[select], 10 * np.log10(psd[select]))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('PSD (dB)')

    slow, slow_fft = myifft(fft, fftfreq)
    axes[0, 0].plot(times, slow)
    slow_psd = np.abs(slow_fft) ** 2
    axes[0, 1].plot(fftfreq[select], 10 * np.log10(slow_psd[select]))

    axes[1, 0].plot(times, slow)
    axes[1, 1].plot(fftfreq[select], 10 * np.log10(psd[select]))
    axes[1, 1].set_xlim(freq_range)

    fig.suptitle('Sound: %d' % j)
    figs.append(fig)

    _, fig = mywavelet(slow)
    fig.suptitle('Sound: %d' % j)
    figs.append(fig)

with PdfPages('multipage_pdf.pdf') as pdf:
    for fig in figs:
        pdf.savefig(fig)

plt.close('all')


# Function for parse sound_array
def parse(sound_array):
    # Prevent change sound_arrcy accidently
    array = sound_array.copy()
    # Make array positive
    array += 1
    # Make array large enough
    array *= 1000
    # Regular array, ravel it and change type
    out = array.ravel().astype(np.short)
    print(out)
    return out


# Function for save wave_array into wavename
def save_wave(wave_array, wavename, framerate=framerate):
    with wave.open(wavename, 'wb') as f:
        # Set number of channels
        f.setnchannels(1)
        # Set number of sampwidth
        f.setsampwidth(2)
        # Set framerate
        f.setframerate(framerate)
        # Write wave_array
        f.writeframes(wave_array)


# For each waveform, save_wave
for j in range(8):
    print(j)
    save_wave(parse(sounds[j]), 'wave_%d.wav' % j)
