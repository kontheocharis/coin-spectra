import matplotlib.pyplot as plot
from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
from typing import List
import warnings

filenames = [
    ['recordings/10c-1.wav', 'recordings/10c-2.wav', 'recordings/10c-3.wav', 'recordings/10c-4.wav', 'recordings/10c-5.wav'],
    ['recordings/20c-1.wav', 'recordings/20c-2.wav', 'recordings/20c-3.wav', 'recordings/20c-4.wav', 'recordings/20c-5.wav'],
    ['recordings/50c-1.wav', 'recordings/50c-2.wav', 'recordings/50c-3.wav', 'recordings/50c-4.wav', 'recordings/50c-5.wav'],
    ['recordings/R1-1.wav',  'recordings/R1-2.wav',  'recordings/R1-3.wav',  'recordings/R1-4.wav',  'recordings/R1-5.wav'],
    ['recordings/R2-1.wav',  'recordings/R2-2.wav',  'recordings/R2-3.wav',  'recordings/R2-4.wav',  'recordings/R2-5.wav'],
    ['recordings/R5-1.wav',  'recordings/R5-2.wav',  'recordings/R5-3.wav',  'recordings/R5-4.wav',  'recordings/R5-5.wav']]

no_of_trials = 5
time_buffer = 6000
sampling_frequency = 0


def main():
    powers = []
    for i, filenames_for_coin in enumerate(filenames):
        coin_name = filenames_for_coin[0].split('/')[1].split('-')[0]

        spectral_densities = []
        powers.append([])
        for j, filename in enumerate(filenames_for_coin):

            sampling_frequency, signal_data = get_data_from_file(filename)
            freq, spectral_density = data_to_spectral_density(sampling_frequency, signal_data)

            spectral_densities.append((freq, spectral_density))

            powers[i].append(spectral_density_to_power(freq, spectral_density))

        freq, spectral_density = np.average(spectral_densities, axis=0)

        # plot_spectral_density(coin_name, freq, spectral_density)
        print_spectral_density(coin_name, freq, spectral_density)

    plot.clf()
    for power in zip(*powers):
        print(power)
        plot.plot(power)
    plot.show()


def get_data_from_file(filename: str) -> (int, List):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', wavfile.WavFileWarning)
        sampling_frequency, signal_data_full = wavfile.read(filename)

    loudest_point = np.argmax(signal_data_full)

    signal_data = signal_data_full[loudest_point - time_buffer:loudest_point + time_buffer]

    return sampling_frequency, signal_data


def data_to_spectral_density(sampling_frequency: int, signal_data: List) -> (List, List):
    freq, spectral_density = signal.welch(signal_data, sampling_frequency)
    return freq, spectral_density


def spectral_density_to_power(freq: List, spectral_density: List) -> int:
    return np.trapz(spectral_density, freq)


def plot_spectral_density(name: str, freq: List, spectral_density: List):
    fig, ax = plot.subplots()
    ax.plot(freq, spectral_density)
    ax.set_title(name + ' Coin Spectral Density')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Spectral Density')
    fig.savefig('figures/' + name + '.png')


def print_spectral_density(name: str, freq, spectral_density: List):
    print('Highest frequency for ' + name + ' coin:')
    print(freq[np.argmax(spectral_density)])
    print(spectral_density_to_power(freq, spectral_density))


if __name__ == "__main__":
    main()
