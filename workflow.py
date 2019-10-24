from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import wave

# Read mat file
fname = 'speechToken.mat'
# Fetch context in mat
mat = sio.loadmat(fname)
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
print(length / framerate)
times = np.array(range(length)) / framerate
for j, s in enumerate(sounds):
    plt.plot(times, s.transpose()/2+j, label=j)
plt.legend()
plt.show()


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
