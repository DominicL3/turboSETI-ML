import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# the astronomy imports
from blimpy import Waterfall

from tqdm import trange
from time import time
import os
import numba # speed up NumPy

from tensorflow.keras.models import load_model
import utils

# used for reading in h5 files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def split_data(data, bins_per_array, enable_numba=True)
    # split 2D data into smaller batches
    start_time = time()
    if data.shape[1] % bins_per_array == 0:
        print("Reshaping array...")
        data = utils.split_reshape(data, bins_per_array)
    elif args.enable_numba:
        print("Splitting array with Numba...")
        data = utils.split_numba(data, bins_per_array)
    else:
        print("Splitting array without Numba optimization...")
        data = utils.split(data, bins_per_array)

    print(f"Split runtime: {np.round(time() - start_time, 4)} seconds")
    return data

def prep_batch_for_prediction(data, bins_per_array, enable_numba=True):
    # normalize each spectrum in an array to 0 median and global stddev to 1
    start_time = time()
    print("Scaling data...")
    if args.enable_numba:
        utils.scale_data_numba(data)
    else:
        utils.scale_data(data)
    print(f"Scaling runtime: {np.round(time() - start_time, 4)} seconds")

    # add channel dimension for Keras tensors (1 channel)
    data = data[..., None]

    return data

def save_to_pdf(pdf_name, t_end, predicted_pulses, pulse_freqs, pulse_probs):
    """Save predicted pulses to PDF, along with their corresponding
    frequencies and prediction probabilities."""

    # compute number of predictions per page by minimizing number of blank axes
    preds_per_page = np.argmin([x - pdf_name % x for x in np.arange(4, 8)])

    fig_height = preds_per_page * 3

    with PdfPages(pdf_name) as pdf:
        for i in trange(len(predicted_pulses)):
            if i % preds_per_page == 0:
                if i != 0:
                    fig_narrowband.tight_layout()
                    pdf.savefig(fig_narrowband, dpi=80)
                    plt.close(fig_narrowband)

                fig_narrowband, ax_narrowband = plt.subplots(nrows=preds_per_page, ncols=2, figsize=(14, fig_height))

            ax = ax_narrowband[i % preds_per_page]

            # extract random array with corresponding freqs and prediction probability
            pulse = predicted_pulses[i]
            freq = pulse_freqs[i]
            prob = np.round(pulse_probs[i], 4)

            # plot spectrogram
            ax[0].imshow(pulse[:, ::-1], aspect='auto', origin='lower',
                        extent=[min(freq), max(freq), 0, t_end])
            ax[0].set(xlabel='freq (MHz)', ylabel='time (s)', title=f"Prediction: {prob}")

            # plot bandpass
            bandpass = np.mean(pulse, axis=0)
            ax[1].plot(freq[::-1], bandpass)
            ax[1].set(xlabel='freq (MHz)', ylabel='power (counts)', title='Bandpass')

            # last plot, so save last page even if incomplete
            if i == len(predicted_pulses) - 1:
                fig_narrowband.tight_layout()
                pdf.savefig(fig_narrowband, dpi=80)
                plt.close(fig_narrowband)

if __name__ == "__main__":
    """
    Parameters
    ---------------
    model_name: str
        Path to trained model used to make prediction. Should be .h5 file
    frb_cand_path: str
        Path to .txt file that contains data about pulses within filterbank file. This
        file should contain columns 'snr','time','samp_idx','dm','filter', and'prim_beam'.
    filterbank_candidate: str
        Path to candidate file to be predicted. Should be .fil file
    NCHAN: int, optional
        Number of frequency channels (default 64) to resize psrchive files to.
    no-FRBcandprob: flag, optional
        Whether or not to save edited FRBcand file containing pulse probabilities.
    FRBcandprob: str, optional
        Path to save FRBcandprob.txt (default is same path as frb_cand_path)
    save_top_candidates: str, optional
        Filename to save pre-processed candidates, just before they are thrown into CNN.
    save_predicted_FRBs: str, optional
        Filename to save every candidate predicted to contain an FRB.
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
    parser.add_argument('-t', '--tchans', type=int, default=None,
                        help='Number of time bins to extract for each sample. If None, use entire integration time.')
    parser.add_argument('-fs', '--f_shift', type=float, default=None,
                        help='Number of frequency channels from start of current frame to begin successive frame. If None, default to no overlap, i.e. f_shift=fchans).')

    parser.add_argument('--thresh', type=float, default=0.5, help='Threshold probability to admit whether example is FRB or RFI.')
    parser.add_argument('--disable_numba', dest='enable_numba', action='store_false',
                        help='Disable numba speed optimizations')

    # options to save outputs
    parser.add_argument('--save_predicted_pulses', type=str, default=None, help='Filename to save all predicted pulses.')

    args = parser.parse_args()

    # load file path
    candidate_file = args.candidate_file
    model_name = args.model_name # either single model or list of models to ensemble predict
    bins_per_array = args.fchans # number of frequency channels per split array

    nbytes_max = args.max_memory * 1e9 # load in at most this many bytes into memory at once

    # load and display summary of given model
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
        print("This file is too large to be loaded in all at once"\
            f"Loading file in {len(freq_windows)} parts, about {args.max_memory} GB each")
        print(f"Each part will contain approximately {freq_bins_per_load} frequency channnels to predict on")
        print(f"Frequency windows for each part (f_start, f_stop): {freq_windows}")

    for test_part in np.arange(len(freq_windows)):
        f_start_max_filesize, f_stop_max_filesize = freq_windows[test_part]
        print(f"Start freq: {f_start_max_filesize}, Stop freq: {f_stop_max_filesize}")

        # load in fil/h5 file into memory
        start_time = time()
        obs = Waterfall(candidate_file, f_start=f_start_max_filesize, f_stop=f_stop_max_filesize, max_load=12)
        print(f"Reading in file took {np.round((time() - start_time)/60, 4)} min")
        obs.freqs = obs.container.populate_freqs()

        # load data and split it so each individual array has bins_per_array freq channels each
        ftdata_test = obs.data[:, 0]
        t_end = obs.header['tsamp'] * obs.n_ints_in_file
        ftdata_test = split_data(ftdata_test, bins_per_array, args.enable_numba)

        # split up frequencies corresponding to data
        freqs_test = obs.freqs.reshape(1, -1)
        freqs_test = split_data(freqs_test, bins_per_array, args.enable_numba)

        # scale candidate arrays and add channel dimension for Keras
        ftdata_test = prep_batch_for_prediction(ftdata_test, bins_per_array)

        # load model and make prediction
        pred_test = model.predict(ftdata_test, verbose=1)[:, 0]

        voted_pulse_probs = pred_test > args.thresh # mask for arrays that were deemed true pulses

        # get paths to predicted pulses and their probabilities
        predicted_pulses = ftdata_test[voted_pulse_probs][:, :, :, 0]
        pulse_freqs = freqs_test[voted_pulse_probs]
        pulse_probs = pred_test[voted_pulse_probs]

        print(f"\nNumber of pulses: {len(voted_pulse_probs)}")

        if len(freq_windows) == 1:
            pdf_name = args.save_predicted_pulses
        else:
            pdf_name = f"{args.save_predicted_pulses.rsplit('.', 1)[0]}_PART{test_part:04d}.pdf"

        save_to_pdf(pdf_name, t_end, predicted_pulses, pulse_freqs, pulse_probs)