import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy import signal
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics_path')
    args = parser.parse_args()
    return args

def interpolation(data):
    return data

def plot_minima(ax, data):
    minima, _ = signal.find_peaks(angles_smoothed * -1.)
    for minimum in minima:
        print('plot minimum {}'.format(minimum))
        ax.axvline(x=minimum, linestyle='--', color='r')

args = parse_args()

angles = np.load(args.metrics_path)
angles = np.nan_to_num(angles, 0.)
fps    = 30.

angles = interpolation(angles)

w = fft(angles)
power = np.abs(w)**2

fft_freq = fftfreq(len(w), 1/fps)
i = fft_freq > 0.

max_power_i = np.argmax(power)
if max_power_i == 0:
    max_power_i = np.argmax(power[1:]) + 1 
fft_freq_at_max_power = fft_freq[max_power_i]
print('Freq at max power: {}'.format(fft_freq_at_max_power))


# Display graphs
fig, ax = plt.subplots(2,1, figsize=[10,7])

# Show the FFT PSD (power density spectrum)
ax[0].plot(fft_freq[i], power[i])
ax[0].set_xlabel('Frequency (1/sec)')
ax[0].set_ylabel('Power Spectrum Density')

# filter out frequencies that are significantly greater than the max frequency
# and calculate the inverse FFT
w_filt = w.copy()
threshold = fft_freq_at_max_power * 1.5 
w_filt[np.abs(fft_freq) > threshold] = 0
angles_smoothed = np.real(ifft(w_filt))

# find location of peaks
minima, _ = signal.find_peaks(angles_smoothed * -1.)
min1 = np.hstack((0,minima))
min2 = np.hstack((minima, len(angles)))
maximums  = [np.max(angles[s:e]) for s, e in zip(min1, min2)]
print(minima)
print(maximums)


# plot original series and smoothed
ax[1].plot(angles)
ax[1].plot(angles_smoothed)
ax[1].set_ylabel('Angles (deg)')
plot_minima(ax[1], angles_smoothed)

plt.show()


