import numpy as np
from numpy.fft import fft, ifft, fftfreq
from scipy import signal
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DESCRIPTION = '''
Plot stride angles generated by gait_metrics.py, run Discrete Fourier Transform (DFT) and plot the implied Gait Cycle.
'''

def dft_analysis(angles):
    '''
    Runs DFT on univariate angles data to locate the true underlying frequency and then extract the maximum
    stride angle at each frequency
    :param angles: input data, np.array
    :return: maxima: The maximum angle for each Gait Cycle
    '''
    angles = np.nan_to_num(angles, 0.)
    fps    = 30.

    angles = interpolation(angles)

    angles_smoothed, fft_freq, power = smooth(angles, fps)

    # find location of peaks
    minima, _ = signal.find_peaks(angles_smoothed * -1.)
    min1 = np.hstack((0,minima))
    min2 = np.hstack((minima, len(angles)))
    maxima  = [np.max(angles[s:e]) for s, e in zip(min1, min2)]
    print(minima)
    print(maxima)

    # Display graphs
    display(angles, angles_smoothed, fft_freq, power)

    return maxima


def display(angles, angles_smoothed, fft_freq, power):
    '''
    Plot 1) Frequency/Power Spectrum density, 2) Original angles data and smoothed data
    :param angles: Original angles data, np.array
    :param angles_smoothed: Smoothed angles data, np.array
    :param fft_freq: Frequency labels for DFT data, np.array
    :param power: Power Spectrum Density for DFT of the angles data
    '''
    fig, ax = plt.subplots(2, 1, figsize=[10, 7])
    # Show the FFT PSD (power density spectrum)
    ax[0].plot(fft_freq, power)
    ax[0].set_xlabel('Frequency (1/sec)')
    ax[0].set_ylabel('Power Spectrum Density')
    # plot original series and smoothed
    ax[1].plot(angles)
    ax[1].plot(angles_smoothed)
    ax[1].set_xlabel('Frame #')
    ax[1].set_ylabel('Angles (deg)')
    plot_minima(ax[1], angles_smoothed)
    plt.show()


def smooth(angles, fps):
    '''
    Run DFT on angles data, locate frequency of max power, then smooth by removing higher frequencies and
    performing Inverse DFT
    :param angles: data to be smoothed, np.array
    :param fps: Sampling rate, float
    :return: angles_smoothed, the smoothed angles data; fft_freq, spectrum frequencies;
                power, the power density spectrum
    '''
    w = fft(angles)
    power = np.abs(w) ** 2
    fft_freq = fftfreq(len(w), 1 / fps)
    i = fft_freq > 0.
    max_power_i = np.argmax(power)
    if max_power_i == 0:
        max_power_i = np.argmax(power[1:]) + 1
    fft_freq_at_max_power = fft_freq[max_power_i]
    print('Freq at max power: {}'.format(fft_freq_at_max_power))
    # filter out frequencies that are significantly greater than the max frequency
    # and calculate the inverse FFT
    w_filt = w.copy()
    threshold = fft_freq_at_max_power * 1.5
    w_filt[np.abs(fft_freq) > threshold] = 0
    angles_smoothed = np.real(ifft(w_filt))
    return angles_smoothed, fft_freq[i], power[i]


def plot_minima(ax, data):
    '''
    Plot dashed vertical line on graph for each minimum value in the data
    :param ax: Matplotlib Axes object
    :param data: Data sequence, np.array
    :return:
    '''
    minima, _ = signal.find_peaks(data * -1.)
    for minimum in minima:
        print('plot minimum {}'.format(minimum))
        ax.axvline(x=minimum, linestyle='--', color='r')

def interpolation(data):
    '''
    TO DO - Angles are set to zero by the gait_metrics program if some of the keypoints are not detected. Can improve
    cycle detection by interpolating missing points
    :param data: data with missing values to be interpolated, np.array
    :return: input data with missing/zero values replaced with interpolated values, np.array
    '''
    return data


def parse_args():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('metrics_path')
    parser.add_argument('--output_append_path', default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    angles = np.load(args.metrics_path)

    maxima = dft_analysis(angles)

    if args.output_append_path is not None:
        f = open(args.output_append_path, 'a+')
        f.write('{}\r\n'.format(np.median(maxima)))
        f.close()

