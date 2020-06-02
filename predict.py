import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# the astronomy imports
from blimpy import Waterfall

from tqdm import trange
from time import time
import os, argparse
import numba # speed up NumPy

# disable eager execution when running with Keras
import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow.keras.models import load_model
import utils

# used for reading in h5 files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def split_data(data, bins_per_array, enable_numba=True):
    # split 2D data into smaller batches
    if data.shape[1] % bins_per_array == 0:
        data = utils.split_reshape(data, bins_per_array)
    elif enable_numba:
        data = utils.split_numba(data, bins_per_array)
    else:
        data = utils.split(data, bins_per_array)
    return data

def prep_batch_for_prediction(data, enable_numba=True):
    # normalize each spectrum in an array to 0 median and global stddev to 1
    if enable_numba:
        utils.scale_data_numba(data)
    else:
        utils.scale_data(data)

    # add channel dimension for Keras tensors (1 channel)
    data = data[..., None]

    return data

def save_to_csv(csv_name, signal_freqs, signal_probs):
    """Save the frequencies and probabilities of predicted signals to CSV.
    Assumes signal_freqs is a 2D array where each row is a frequency slice
    belonging to a predicted signal, and that signal_probs is a 1D array."""

    hdr = "Min freq (MHz), Max freq (MHz), Probability"
    min_freqs = np.min(signal_freqs, axis=1)
    max_freqs = np.max(signal_freqs, axis=1)

    data = np.array([min_freqs, max_freqs, signal_probs]).reshape(-1, 3)

    if os.path.isfile(csv_name):
        loaded_data = np.loadtxt(csv_name, skiprows=1)
        data = np.concatenate([loaded_data, data], axis=0)

    np.savetxt(csv_name, data, fmt='%-10.5f', header=hdr)


def save_to_pdf(pdf_name, t_end, predicted_signals, signal_freqs, signal_probs):
    """Save predicted signals to PDF, along with their corresponding
    frequencies and prediction probabilities."""

    # compute number of predictions per page by minimizing number of blank axes
    preds_per_page = 4 + np.argmax([(len(predicted_signals) % x) / x for x in np.arange(4, 9)])
    fig_height = preds_per_page * 3

    with PdfPages(pdf_name) as pdf:
        for i in trange(len(predicted_signals)):
            if i % preds_per_page == 0:
                if i != 0:
                    fig_narrowband.tight_layout()
                    pdf.savefig(fig_narrowband, dpi=80)
                    plt.close(fig_narrowband)

                fig_narrowband, ax_narrowband = plt.subplots(nrows=preds_per_page, ncols=2, figsize=(14, fig_height))

            ax = ax_narrowband[i % preds_per_page]

            # extract random array with corresponding freqs and prediction probability
            signal = predicted_signals[i]
            freq = signal_freqs[i]
            prob = np.round(signal_probs[i], 4)

            # plot spectrogram
            ax[0].imshow(signal[:, ::-1], aspect='auto', origin='lower',
                        extent=[min(freq), max(freq), 0, t_end])
            ax[0].set(xlabel='freq (MHz)', ylabel='time (s)', title=f"Prediction: {prob}")

            # plot spectrum
            spectrum = np.mean(signal, axis=0)
            ax[1].plot(freq[::-1], spectrum)
            ax[1].set(xlabel='freq (MHz)', ylabel='power (counts)', title='Frequency Spectrum')

            # last plot, so save last page even if incomplete
            if i == len(predicted_signals) - 1:
                fig_narrowband.tight_layout()
                pdf.savefig(fig_narrowband, dpi=80)
                plt.close(fig_narrowband)

if __name__ == "__main__":
    """
    Parameters
    ---------------
    candidate_file: str
        Path to candidate file to be predicted. Should be .fil or .h5 file.
    model_name: str
        Path to trained model used to make prediction. Should be a Keras .h5 file.
    fchans: int, optional
        Number of frequency channels (default 1024) to extract from each array.
    save_predicted_signals: str, optional
        Filename to save every candidate predicted to contain a signal.
    """

    # Read command line arguments
    parser = argparse.ArgumentParser()

    # main arguments needed for prediction
    parser.add_argument('candidate_file', type=str, help='Path to fil/h5 file to be predicted on.')
    parser.add_argument('model_name', type=str, help='Path to trained model used to make prediction.')

    # can set if pickle files are already in directory to avoid having to redo extraction
    parser.add_argument('-mem', '--max_memory', type=float, default=1,
                        help='Maximum amount of memory (GB) to load in from file at one time.')

    # control number of freq/time channels from each array.
    # Default value of None means all time channels are used.
    parser.add_argument('-f', '--fchans', type=int, default=1024,
                        help='Number of frequency channels to extract for each sample in candidate file.')
    parser.add_argument('-fs', '--f_shift', type=float, default=None,
                        help='Number of frequency channels from start of current frame to begin successive frame. If None, default to no overlap, i.e. f_shift=fchans).')

    parser.add_argument('-p', '--thresh', type=float, default=0.5, help='Threshold probability to admit whether example is FRB or RFI.')
    parser.add_argument('--disable_numba', dest='enable_numba', action='store_false',
                        help='Disable numba speed optimizations')

    # options to save outputs
    parser.add_argument('-csv', '--csv_name', type=str, default='predictions.csv', help='Filename (csv) to save all predicted signals.')
    parser.add_argument('-pdf', '--save_pdf', type=str, default=None, help='Filename (pdf) to save all predicted signals.')

    args = parser.parse_args()

    # update runtime as prediction moves along
    script_start_time = time()

    # load file path
    candidate_file = args.candidate_file
    model_name = args.model_name # either single model or list of models to ensemble predict
    bins_per_array = args.fchans # number of frequency channels per split array

    nbytes_max = args.max_memory * 1e9 # load in at most this many bytes into memory at once

    # load model and display summary
    model = load_model(model_name, compile=True)
    print(model.summary())

    # get fil/h5 file header
    obs = Waterfall(candidate_file, load_data=False)

    # range of spectrum
    f_start = obs.container.f_start
    f_stop = obs.container.f_stop

    freq_bins_per_load = (f_stop - f_start) / (obs.container.file_size_bytes) * nbytes_max

    freq_windows = [(f_start + i * freq_bins_per_load, f_start + (i+1) * freq_bins_per_load)
                    for i in np.arange(np.ceil(obs.container.file_size_bytes / nbytes_max))]

    if len(freq_windows) > 1:
        print("\nThis file is too large to be loaded in all at once. "\
            f"Loading file in {len(freq_windows)} parts, about {args.max_memory} GB each")
        print(f"Each part will contain approximately {freq_bins_per_load} frequency channnels to predict on")
        print(f"Frequency windows for each part (f_start, f_stop): {freq_windows}")

    total_signals = 0 # running total of number of signals in entire file
    for test_part in np.arange(1, len(freq_windows) + 1):
        print(f"\nAnalyzing part {test_part} / {len(freq_windows)}:")
        f_start_max_filesize, f_stop_max_filesize = freq_windows[test_part - 1]
        print(f"Loading data from f_start={f_start_max_filesize} to f_stop={f_stop_max_filesize}...")

        # load in fil/h5 file into memory
        start_time = time()
        obs = Waterfall(candidate_file, f_start=f_start_max_filesize, f_stop=f_stop_max_filesize, max_load=12)
        print(f"Loading data took {np.round((time() - start_time)/60, 4)} min\n")
        obs.freqs = obs.container.populate_freqs()

        # copy data
        print("Copying data...")
        if args.enable_numba:
            ftdata_test = utils.copy_2d_data_numba(obs.data[:, 0])
        else:
            ftdata_test = utils.copy_2d_data(obs.data[:, 0])
        print(f"Copying data took {np.round(time() - start_time, 4)} seconds\n")

        obs.data = None # free up memory since obs.data isn't needed anymore

        # split 2D array into 3D array so each individual array has bins_per_array freq channels each
        print("Splitting array...")
        start_time = time()
        ftdata_test = split_data(ftdata_test, bins_per_array, args.enable_numba)
        print(f"Split runtime: {np.round(time() - start_time, 4)} seconds\n")

        # split up frequencies corresponding to data
        freqs_test = obs.freqs.reshape(1, -1)
        freqs_test = split_data(freqs_test, bins_per_array, args.enable_numba)
        freqs_test = freqs_test[:, 0]

        # scale candidate arrays and add channel dimension for Keras
        start_time = time()
        print("Scaling data and preparing batch for prediction...")
        ftdata_test = prep_batch_for_prediction(ftdata_test, args.enable_numba)
        print(f"Scaling runtime: {np.round(time() - start_time, 4)} seconds\n")

        # load model and make prediction
        pred_test = model.predict(ftdata_test, verbose=1)[:, 0]

        voted_signal_probs = pred_test > args.thresh # mask for arrays that were deemed true signals

        # get paths to predicted signals and their probabilities
        predicted_signals = ftdata_test[voted_signal_probs][:, :, :, 0]
        signal_freqs = freqs_test[voted_signal_probs]
        signal_probs = pred_test[voted_signal_probs]

        # count number of predicted signals in file and add to running total
        num_signals_in_file = np.sum(voted_signal_probs)
        total_signals += num_signals_in_file

        print(f"\nNumber of signals in part: {num_signals_in_file}")
        print(f"Total signals found: {total_signals}")

        print(f"\nStoring info on {len(predicted_signals)} predicted candidates to {args.csv_name}")
        save_to_csv(args.csv_name, signal_freqs, signal_probs)

        if args.save_pdf:
            # break pdf into parts if > 1 chunks are extracted
            if len(freq_windows) == 1:
                pdf_name = args.save_pdf
            else:
                pdf_name = f"{args.save_pdf.rsplit('.', 1)[0]}_PART{test_part:04d}.pdf"

            # compute observation time in secondss
            t_end = obs.header['tsamp'] * obs.n_ints_in_file

            print(f"Saving images of {len(predicted_signals)} signals to {pdf_name}")
            save_to_pdf(pdf_name, t_end, predicted_signals, signal_freqs, signal_probs)

        print(f"Elapsed runtime: {np.round((time() - script_start_time) / 60, 2)} minutes")

    print(f"Total runtime: {np.round((time() - script_start_time) / 60, 2)} minutes")