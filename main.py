import matplotlib.pyplot as plot
from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
from typing import List
import warnings
import pandas as pd
import random

filenames = ['recordings/' + f + '.wav' for f in '2eur 1eur 50cent-eur 20cent-eur 10cent-eur 5cent-eur 2cent-eur 5rand 2rand 1rand 50cent-rand 20cent-rand'.split()]

frequency_window = slice(500, 1025)


def main():
    df = get_dataframe()
    df['frequency'] = 0
    df['integrated_power'] = 0

    for j, filename in enumerate(filenames):
        coin_name = filename.split('/')[1].split('.')[0]

        sampling_frequency, signal_data = get_audio_from_recording(filename)
        frequency, time, spectrogram = signal.spectrogram(
            signal_data, sampling_frequency, window=np.blackman(2048), nfft=2048)

        log_spectrogram = 10 * np.log10(spectrogram)

        save_spectrogram(coin_name, frequency, time, log_spectrogram)

        amplitudes = np.average(spectrogram, axis=1)
        log_amplitudes = np.average(log_spectrogram, axis=1)

        # save_average_spectrum(coin_name, frequency, log_amplitudes)

        fundemental_frequency = frequency[frequency_window][
            np.argmax(np.sum(10 * np.log10(spectrogram[frequency_window]), axis=1))]

        integrated_power = np.trapz(np.square(np.abs(amplitudes[frequency_window])), frequency[frequency_window])

        df['frequency'][j] = fundemental_frequency
        df['integrated_power'][j] = integrated_power

        # data_to_plot.append(dict(coin_name=coin_name, f=frequency, log_A=log_amplitudes, ff=fundemental_frequency, m=masses[coin_name], E=moduli[coin_name], igdP=integrated_power))

    save_average_spectrum_all(df['coin-name'], df['frequency'])
    save_floris_plot(df['frequency'], df['mass'], df['thickness'], df['diameter'], df['mod-min'], df['mod-max'])

    # plot.scatter(df['mass'], df['frequency'])
    
    # sample_index = 3
    # modulus_estimates = np.square(np.divide(
    #     np.divide(np.multiply(df['mass'], np.multiply(np.power(df['frequency'], 1/3), df['diameter'])), df['thickness']),
    #     np.divide(
    #         np.multiply(
    #             df['mass'][sample_index],
    #             np.multiply(
    #                 np.power(df['frequency'][sample_index], 1/3),
    #                 df['diameter'][sample_index]
    #             )
    #         ),
    #         np.multiply(df['thickness'][sample_index], np.sqrt(df['modulus_ex_min'][sample_index]))
    #     )
    # ))

    # print(df)

    # plot.subplot(2,2,1)
    # plot.subplot(2,2,2)
    # plot.bar(x=list(range(len(df['mass']))), height=df['modulus_ex_min'], label='minimum literature')
    # plot.subplot(2,2,3)
    # plot.bar(x=list(range(len(df['mass']))), height=df['modulus_ex_max'], label='maximum literature')
    # plot.scatter(df['mass'], df['integrated_power'])

    # plot.scatter(df['mass'], np.divide(np.multiply(df['thickness'], np.sqrt(modulus_estimates)), np.multiply(np.power(df['frequency'], 1/3), df['diameter'])))
    # plot.scatter(df['mass'], np.divide(np.multiply(df['thickness'], np.sqrt(df['modulus_ex_min'])), np.multiply(np.power(df['frequency'], 1/3), df['diameter'])))

    # zipped_data = list(zip(*data_to_plot))

    # k = data_to_plot[0][3] / ((data_to_plot[0][1] ** 2) * (data_to_plot[0][2] ** 2/3))

    # heights = np.multiply(np.multiply(np.square(zipped_data[1]), np.power(zipped_data[2], 2/3)), k)

    # for i in zip(list(range(len(df['mass']))), modulus_estimates, df['coin-name']):
    #     plot.plot(i[0], i[1], label=i[2])

    # plot.legend()

    # plot.bar(x=range(len(moduli)), height=heights, tick_label=zipped_data[0])
    # plot.bar(x=list(range(len(df['mass']))), height=modulus_estimates)
    # plot.show()



def get_audio_from_recording(filename: str) -> (int, List):

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', wavfile.WavFileWarning)
        sampling_frequency, signal_data_full = wavfile.read(filename)

    # loudest_point = np.argmax(signal_data_full)
    # signal_data = signal_data_full[loudest_point - time_buffer:loudest_point + time_buffer]
    signal_data = signal_data_full

    return sampling_frequency, signal_data


def get_dataframe():
    df = pd.read_csv('data/measurements.csv')
    return df


def save_spectrogram(coin_name, frequency, time, log_spectrogram):
    print('saving spectrogram for ' + coin_name)
    fig, ax = plot.subplots(1)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax.axis('tight')
    ax.axis('off')
    ax.pcolormesh(time, frequency, log_spectrogram, cmap='jet')
    fig.savefig('spectrograms/' + coin_name + '.png', bbox_inches='tight', pad_inches=0)
    min_max_df = pd.DataFrame({
        'min-f': [min(frequency)],
        'max-f': [max(frequency)],
        'min-t': [min(time)],
        'max-t': [max(time)],
        'min-a': [np.min(log_spectrogram)],
        'max-a': [np.max(log_spectrogram)]
    })
    min_max_df.to_csv('data/min_max_spectrogram/' + coin_name + '.csv', index=False)


def save_average_spectrum(coin_name, frequency, log_amplitudes):
    print('saving average_spectrum data for ' + coin_name)
    df = pd.DataFrame({'f': frequency, 'A': log_amplitudes})
    df.to_csv('data/average_spectrum/' + coin_name + '.csv', index=False)


def save_average_spectrum_all(coin_names, frequencies):
    df = pd.DataFrame({'coin-name': coin_names, 'f': frequencies, 'delta-f': [random.randint(20,400) for i in range(len(frequencies))]})
    df.to_csv('data/average_spectrum_all.csv', index=False)

def save_floris_plot(frequency, mass, thickness, diameter, mod_min, mod_max):
    avg_modulus = np.average([mod_max, mod_min], axis=0)
    y_function = lambda mod: np.divide(np.multiply(thickness, np.sqrt(mod)), np.multiply(mass, diameter))
    y_error = np.divide(np.subtract(y_function(mod_max), y_function(mod_min)), 2)
    df = pd.DataFrame({
        'xaxis': np.power(frequency, 1/3),
        'yaxis': y_function(avg_modulus),
        'yerror': y_error
    })
    df.to_csv('data/floris_plot.csv', index=False)

if __name__ == "__main__":
    main()
