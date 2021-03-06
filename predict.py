import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# the astronomy imports
from blimpy import Waterfall

from tqdm import tqdm, trange
from time import time
import os, sys, argparse

# modules for performance increases
import numba
from waterfall_loader import ThreadedWaterfallLoader
import gc # garbage collect every now and then
import multiprocessing as mp

from tensorflow.keras.models import load_model, Model
import utils

# used for reading in h5 files
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

def split_data(data, bins_per_array, f_shift, enable_numba=True):
    # split 2D data into smaller batches
    if enable_numba:
        data = utils.split_numba(data, bins_per_array, f_shift)
    else:
        data = utils.split(data, bins_per_array, f_shift)
    return data

def prep_batch_for_prediction(data, enable_numba=True):
    # normalize each spectrum in an array to 0 median and global stddev to 1
    if enable_numba:
        utils.scale_data_numba(data)
    else:
        utils.scale_data(data)

    # zero pad array when max pooling may remove all dimensions
    if data.shape[1] < 16:
        data = utils.zero_pad_array(data)

    data = data[..., None]
    return data

def save_to_csv(csv_name, signal_freqs, signal_probs, drift_rates_ML, drift_rates_regression=None):
    """Save the frequencies and probabilities of predicted signals to CSV
    as well as the predicted drift rate of the signal. Assumes signal_freqs
    is a 2D array where each row is a frequency slice belonging to a predicted
    signal, and that signal_probs is a 1D array."""

    # create and fill DataFrame
    csv_data = pd.DataFrame(dtype=signal_freqs.dtype)
    csv_data["Min freq (MHz)"] = np.min(signal_freqs, axis=1) # frequency window of every predicted signal
    csv_data["Max freq (MHz)"] = np.max(signal_freqs, axis=1)
    csv_data["Probability"] = signal_probs
    csv_data["ML Drift Rate (Hz/s)"] = drift_rates_ML

    # add extra column for regression transform drift rates if enabled
    # should be array with columns (drift_pred, lower, upper) signifying
    # the confidence intervals of the drift rate prediction
    if drift_rates_regression is not None:
        if drift_rates_regression.ndim == 1:
            csv_data["Regression Drift Rate (Hz/s)"] = drift_rates_regression
        elif drift_rates_regression.shape[1] == 3:
            df_drift = pd.DataFrame(data=drift_rates_regression, columns=["Regression Drift Rate (Hz/s)",
                                        "Lower Drift Rate (99%)", "Upper Drift Rate (99%)"])
        else:
            raise ValueError("drift_rates_regression needs to be array with either 1 or 3 columns.")

        csv_data = pd.concat([csv_data, df_drift], axis=1)

    # append predictions to file if it already exists
    if os.path.isfile(csv_name):
        loaded_data = pd.read_csv(csv_name, sep='\t')
        csv_data = loaded_data.append(csv_data, ignore_index=True)

    csv_data.sort_values("Min freq (MHz)", inplace=True) # sort lowest to highest frequency
    csv_data.to_csv(csv_name, sep='\t', index=False, float_format='%-10.6f')

def save_to_pdf(pdf_name, t_end, predicted_signals, signal_freqs, signal_probs,
                    drift_rates_ML, drift_rates_regression=None):
    """Save predicted signals to PDF, along with their corresponding
    frequencies, prediction probabilities, and drift rates."""

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
            prob = signal_probs[i]
            drift_ML = drift_rates_ML[i]

            # what to title frequency spectrum based on whether regression drift rates are passed in
            if drift_rates_regression is not None:
                drift_regression = drift_rates_regression[i]
                spec_title = f'Drift rate (ML): {drift_ML:.4f} Hz/s, Drift rate (regression): {drift_regression:.4f} Hz/s'
            else:
                spec_title = f'Drift rate (ML): {drift_ML:.4f} Hz/s'

            # plot spectrogram
            f_start, f_stop = np.min(freq), np.max(freq) # get start and end freqs
            ax[0].imshow(signal, aspect='auto', origin='lower',
                        extent=[f_start, f_stop, 0, t_end])
            ax[0].set(title=f"Index: {i}, Prediction: {prob:.4f}, Start/End Freqs (MHz): {f_start:.4f}-{f_stop:.4f}",
                        xlabel='freq (MHz)', ylabel='time (s)')

            # plot spectrum
            spectrum = np.mean(signal, axis=0)
            ax[1].plot(freq, spectrum)
            ax[1].set(xlabel='freq (MHz)', ylabel='power (counts)', title=spec_title)

            # last plot, so save last page even if incomplete
            if i == len(predicted_signals) - 1:
                fig_narrowband.tight_layout()
                pdf.savefig(fig_narrowband, dpi=80)
                plt.close(fig_narrowband)

def find_signals(wt_loader, model, bins_per_array=1024, f_shift=None, threshold=0.5, enable_numba=True):
    # load in fil/h5 file into memory
    start_time = time()
    freqs_test, ftdata_test = wt_loader.get_observation() # grab 2D data and frequencies
    print(f"Loading data took {(time() - start_time)/60:.4f} min\n")

    # split 2D array into 3D array so each individual array has bins_per_array freq channels each
    print("Splitting array...")
    start_time = time()
    ftdata_test = split_data(ftdata_test, bins_per_array, f_shift, enable_numba=enable_numba)
    print(f"Split runtime: {time() - start_time:.4f} seconds")
    print(f"Split array has shape {ftdata_test.shape}\n")

    # split up frequencies corresponding to data
    freqs_test = split_data(freqs_test.reshape(1, -1), bins_per_array, f_shift, enable_numba=enable_numba)
    freqs_test = freqs_test[:, 0] # remove extra dimension created to split_data

    # delete large variable and run garbage collection
    gc.collect()

    # scale candidate arrays and add channel dimension for Keras
    start_time = time()
    print("Scaling data and preparing batch for prediction...")
    ftdata_test = prep_batch_for_prediction(ftdata_test, enable_numba=enable_numba)
    print(f"Scaling runtime: {time() - start_time:.4f} seconds\n")

    # predict class and drift rate with model
    print("Predicting with model...")
    pred_test, slopes_test = model.predict(ftdata_test, verbose=1)
    pred_test, slopes_test = pred_test.flatten(), slopes_test.flatten()

    voted_signal_probs = pred_test > threshold # mask for arrays that were deemed true signals

    # count number of predicted signals in file and add to running total
    num_signals_in_file = np.sum(voted_signal_probs)
    print(f"\nNumber of signals found: {num_signals_in_file}")

    # get paths to predicted signals and their probabilities
    predicted_signals = ftdata_test[voted_signal_probs, :, :, 0]
    signal_freqs = freqs_test[voted_signal_probs] # freqs for predicted signals
    signal_probs = pred_test[voted_signal_probs] # probabilities of predictions for signals
    drift_rates_ML = utils.get_driftRate_from_slope(slopes_test[voted_signal_probs], df, dt)

    return predicted_signals, signal_freqs, signal_probs, drift_rates_ML

def compute_slopes_regression(predicted_signals, num_cores=0):
    # use multiprocessing only if overhead is worth it (more signals than cores)
    if num_cores > 0:
        print(f"Running in parallel with {num_cores} cores")
        # put predicted signals into shared memory for multiprocessing
        # shared = mp.RawArray('d', predicted_signals.size)
        # shared_preds = np.frombuffer(shared).reshape(predicted_signals.shape)
        # np.copyto(shared_preds, predicted_signals)
        with mp.Pool(num_cores) as pool:
            try:
                regression_slopes = np.array([slope for slope in tqdm(pool.imap(utils.regression_slope, predicted_signals), total=len(predicted_signals))])
            except KeyboardInterrupt:
                print("Parent received KeyboardInterrupt, exiting...")
                pool.terminate()
                pool.join()
                sys.exit()
    else:
        regression_slopes = np.array([utils.regression_slope(data) for data in tqdm(predicted_signals)])

    return regression_slopes

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
    parser.add_argument('-fs', '--f_shift', type=int, default=None,
                        help='Number of frequency channels from start of current frame to begin successive frame. If None, default to no overlap, i.e. f_shift=fchans).')

    parser.add_argument('-p', '--thresh', type=float, default=0.5, help='Threshold probability to admit whether example is FRB or RFI.')
    parser.add_argument('-cores', '--num_cores', type=int, default=0,
                        help='Number of cores to use for multiprocessing. Defaults to using all available processors. Set to 0 to disable.')
    parser.add_argument('--disable_numba', dest='enable_numba', action='store_false', help='Disable numba speed optimizations')

    # options to save outputs
    parser.add_argument('--no_regression', dest='include_regression_drift', action='store_false',
                        help='Do not compute drift rate from traditional regression transforms.')
    parser.add_argument('-csv', '--csv_name', type=str, default='predictions.csv', help='Filename (csv) to save all predicted signals.')
    parser.add_argument('-pdf', '--save_pdf', type=str, default=None, help='Filename (pdf) to save all predicted signals.')

    args = parser.parse_args()

    # update runtime as prediction moves along
    script_start_time = time()

    # load file path
    candidate_file = args.candidate_file
    model_name = args.model_name # either single model or list of models to ensemble predict
    bins_per_array = args.fchans # number of frequency channels per split array
    f_shift = bins_per_array if args.f_shift is None else args.f_shift # default to no overlapping arrays

    nbytes_max = args.max_memory * 1e9 # load in at most this many bytes into memory at once
    nbytes_per_part = nbytes_max / 2 # half of memory for current array, other half for array waiting on queue

    # load model and display summary
    model = load_model(model_name, compile=False)
    print(model.summary())

    # get fil/h5 file header
    obs = Waterfall(candidate_file, load_data=False)
    # TODO: move df, dt, and t_end into ThreadedWaterfallLoader so as not to rely on global vars
    df = obs.header['foff'] * 1e6 # sampling frequency, converted from MHz to Hz
    dt = obs.header['tsamp'] # sampling time in seconds
    t_end = obs.header['tsamp'] * obs.n_ints_in_file # observation time in seconds

    # range of entire spectrum
    f_start = obs.container.f_start
    f_stop = obs.container.f_stop

    # compute how many times we need to split up filterbank file under memory restrictions
    num_parts = np.ceil(obs.file_size_bytes / nbytes_per_part * (bins_per_array/f_shift))
    freqs_per_load = (f_stop - f_start) / num_parts

    freq_windows = [(f_start + i * freqs_per_load, f_start + (i+1) * freqs_per_load)
                            for i in np.arange(num_parts)]

    if len(freq_windows) > 1:
        print("\nThis file is too large to be loaded in all at once. "\
            f"Loading file in {len(freq_windows)} parts, about {args.max_memory / 2} GB each")
        print(f"Each part will contain approximately {freqs_per_load} MHz to predict on")
        print(f"Frequency windows for each part (f_start, f_stop): {freq_windows}")

    # begin loading in data
    wt_loader = ThreadedWaterfallLoader(candidate_file, freq_windows, max_memory=args.max_memory)
    wt_loader.start()

    total_signals = 0 # running total of number of signals in entire file
    if os.path.isfile(args.csv_name): # delete old file if it exists
            os.remove(args.csv_name)

    # iterate through all parts of the fil/h5 file
    for test_part in np.arange(1, len(freq_windows) + 1):
        part_start_time = time()
        print(f"\nANALYZING PART {test_part} / {len(freq_windows)}:")
        f_start_max_filesize, f_stop_max_filesize = freq_windows[test_part - 1]
        print(f"Loading data from f_start={f_start_max_filesize} MHz to f_stop={f_stop_max_filesize} MHz...")

        predicted_signals, signal_freqs, signal_probs, drift_rates_ML = find_signals(wt_loader, model, bins_per_array=bins_per_array, f_shift=f_shift,
                                                                                        threshold=args.thresh, enable_numba=args.enable_numba)
        gc.collect()

        num_signals_in_file = len(predicted_signals)
        total_signals += num_signals_in_file

        if args.include_regression_drift:
            start_time = time()
            print("\nComputing drift rate from linear regression methods...")

            # compute drift rate for all predicted signals from regression methods
            regression_slopes = compute_slopes_regression(predicted_signals, num_cores=args.num_cores)
            drift_rates_regression = utils.get_driftRate_from_slope(regression_slopes, df, dt)

            print(f"Finished estimating drift rates in {time() - start_time:.2f} seconds"\
                    f" | {num_signals_in_file/(time() - start_time):.2f}it/s")
        else:
            drift_rates_regression = None

        print(f"\nStoring info on {num_signals_in_file} predicted candidates to {args.csv_name}")

        # save data to csv and/or pdf only if at least one signal was found
        if num_signals_in_file > 0:
            # save frequencies and prediction probabilities to csv
            save_to_csv(args.csv_name, signal_freqs, signal_probs,
                            drift_rates_ML, drift_rates_regression)

            if args.save_pdf:
                # break pdf into parts if > 1 chunks are extracted
                if wt_loader.q_freqs.empty(): # put all in one PDF if no more chunks to extract
                    pdf_name = args.save_pdf
                else:
                    pdf_name = f"{args.save_pdf.rsplit('.', 1)[0]}_PART{test_part:04d}.pdf"

                print(f"Saving images of {num_signals_in_file} signals to {pdf_name}")
                save_to_pdf(pdf_name, t_end, predicted_signals, signal_freqs, signal_probs,
                                drift_rates_ML, drift_rates_regression)

        print(f"\nTotal signals found: {total_signals}")
        print(f"Analyzing part {test_part} took {(time() - part_start_time)/60:.2f} minutes")
        print(f"Elapsed runtime: {(time() - script_start_time)/60:.2f} minutes")

        # section break
        print("\n" + "==============================" * 3)

    # stop thread
    wt_loader.stop()
    print("\nDONE!")