from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import wave

fname = 'speechToken.mat'
mat = sio.loadmat(fname)
pprint(mat)
print(mat['tokenS'].shape)

sounds = [e[0] for e in mat['tokenS']]
print(sounds)
num = len(sounds)
length = len(sounds[0][0])
print(num, length)

framerate = 44100
times = np.array(range(length)) / framerate
for j, s in enumerate(sounds):
    plt.plot(times, s.transpose()/2+j, label=j)
plt.legend()
plt.show()


def parse(sound_array):
    return sound_array.ravel().astype(np.float)


def save_wave(wave_array, wavename, framerate=framerate):
    with wave.open(wavename, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(framerate)
        f.writeframes(wave_array)


for j in range(8):
    print(j)
    save_wave(parse(sounds[j]), 'wave_%d.wav' % j)
