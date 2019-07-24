import matplotlib.pyplot as plot
from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
from typing import List
import warnings

# filenames = [
#     ['recordings/10c-1.wav', 'recordings/10c-2.wav', 'recordings/10c-3.wav', 'recordings/10c-4.wav', 'recordings/10c-5.wav'],
#     ['recordings/20c-1.wav', 'recordings/20c-2.wav', 'recordings/20c-3.wav', 'recordings/20c-4.wav', 'recordings/20c-5.wav'],
#     ['recordings/50c-1.wav', 'recordings/50c-2.wav', 'recordings/50c-3.wav', 'recordings/50c-4.wav', 'recordings/50c-5.wav'],
#     ['recordings/R1-1.wav',  'recordings/R1-2.wav',  'recordings/R1-3.wav',  'recordings/R1-4.wav',  'recordings/R1-5.wav'],
#     ['recordings/R2-1.wav',  'recordings/R2-2.wav',  'recordings/R2-3.wav',  'recordings/R2-4.wav',  'recordings/R2-5.wav'],
#     ['recordings/R5-1.wav',  'recordings/R5-2.wav',  'recordings/R5-3.wav',  'recordings/R5-4.wav',  'recordings/R5-5.wav']]

filenames = [
    'recordings/10cent-eur.wav',
    'recordings/1eur.wav',
    'recordings/1rand.wav',
    'recordings/20cent-eur.wav',
    'recordings/20cent-rand.wav',
    'recordings/2cent-eur.wav',
    'recordings/2eur.wav',
    'recordings/2rand.wav',
    'recordings/50cent-eur.wav',
    'recordings/50cent-rand.wav',
    'recordings/5cent-eur.wav',
    'recordings/5rand.wav']

# no_of_trials = 5
# time_buffer = 6000

frequency_window = slice(500, 1025)


def main():
    data_to_plot = []
    for j, filename in enumerate(filenames):
        coin_name = filename.split('/')[1].split('.')[0]

        sampling_frequency, signal_data = get_data_from_file(filename)
        frequency, time, spectrogram = signal.spectrogram(
            signal_data, sampling_frequency, window=np.blackman(2048), nfft=2048)

        fundemental_frequency = frequency[frequency_window][
            np.argmax(np.sum(10 * np.log10(spectrogram[frequency_window]), axis=1))]
        data_to_plot.append((coin_name, fundemental_frequency))

    zipped_data = list(zip(*data_to_plot))
    labels = zipped_data[1]

    plot.bar(x=range(len(zipped_data[0])), height=zipped_data[1], tick_label=zipped_data[0])

    for i, v in enumerate(labels):
        plot.text(i-.25, v/labels[i]+100, labels[i], color='white')
    plot.show()


def get_data_from_file(filename: str) -> (int, List):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', wavfile.WavFileWarning)
        sampling_frequency, signal_data_full = wavfile.read(filename)

    # loudest_point = np.argmax(signal_data_full)
    # signal_data = signal_data_full[loudest_point - time_buffer:loudest_point + time_buffer]
    signal_data = signal_data_full

    return sampling_frequency, signal_data


if __name__ == "__main__":
    main()
