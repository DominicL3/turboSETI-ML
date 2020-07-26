import numpy as np
from sklearn.model_selection import train_test_split

import setigen as stg

import tqdm
from time import time
import os, argparse

import multiprocessing as mp # for parallel code execution

# neural net imports
from tensorflow.keras.models import load_model
from model import construct_model, fit_model

import generate_dataset, utils # for sampling parameter distributions

"""
Train a Keras model to do binary classification on simulated pulses
vs. background RFI and save the best model from training. Exit automatically
if validation loss doesn't improve after a certain number of epochs.

Takes in a .fil file to use as background RFI and uses setigen to simulate
narrowband signals with randomly generated signal properties.

@source Liam Connor (https://github.com/liamconnor/single_pulse_ml)
@source Bryan Brzycki (https://github.com/bbrzycki/setigen)
"""

def add_rfi(frame, SNRmin=10, SNRmax=20, min_drift=-5, max_drift=5, min_width=5, max_width=30):
    fchans = frame.fchans

    # let true pulse begin in middle 50% of array and randomize drift rate
    start_index = np.random.randint(0, fchans)
    drift_rate = np.random.uniform(min_drift, max_drift)

    random_SNR = np.random.uniform(SNRmin, SNRmax)
    width = np.random.uniform(min_width, max_width)

    frame.add_signal(stg.paths.choppy_rfi_path(frame.get_frequency(start_index), drift_rate, fchans, spread_type='normal'),
                 stg.constant_t_profile(level=frame.get_intensity(snr=random_SNR)),
                 stg.gaussian_f_profile(width=width),
                 stg.constant_bp_profile(level=1))

def simulate_signal(frame, SNRmin=10, SNRmax=20, min_drift=-5, max_drift=5,
                    min_width=5, max_width=30, add_to_frame=True):
    """Generate dataset, taken from setigen docs (advanced topics)."""
    fchans = frame.fchans

    # let true pulse begin in middle 50% of array and randomize drift rate
    start_index = np.random.randint(0.25 * fchans, 0.75 * fchans)
    drift_rate = np.random.uniform(min_drift, max_drift)

    # sample SNR and frequency profile randomly
    random_SNR = np.random.uniform(SNRmin, SNRmax)
    width = np.random.uniform(min_width, max_width)
    f_profile_type = np.random.choice(['box', 'gaussian', 'lorentzian', 'voigt'])

    # add metadata to each frame for bookkeeping purposes
    signal_props = {
        'start_index': start_index,
        'drift_rate': drift_rate,
        'snr': random_SNR,
        'width': width,
        'f_profile_type': f_profile_type
    }
    frame.add_metadata(signal_props)

    if add_to_frame:
        # add signal to background
        signal = frame.add_constant_signal(f_start=frame.get_frequency(start_index),
                                            drift_rate=drift_rate,
                                            level=frame.get_intensity(snr=random_SNR),
                                            width=width,
                                            f_profile_type=f_profile_type)

def parse_frame_args(fchans, tchans, df, dt, min_freq, max_freq, rfi_prob, means, stddevs, mins):
    """Takes arguments for frame and condenses it into one tuple
    to be passed into make_artificial_frame(). Used for multiprocessing,
    since parallel processes only take in one argument."""

    frame_args = [fchans, tchans, df, dt, min_freq, max_freq, rfi_prob]
    distribution_args = [means, stddevs, mins]

    return (frame_args, distribution_args)

def make_artificial_frame(simulation_args):
    # unpack arguments
    frame_args, distribution_args = simulation_args
    fchans, tchans, df, dt, min_freq, max_freq, rfi_prob = frame_args
    means, stddevs, mins = distribution_args

    # randomly sample frequency at end of array
    fch1 = np.random.uniform(min_freq + df * fchans, max_freq)

    # create setigen frame with frame args
    frame = stg.Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1)

    # add chi-squared noise to frame using distribution args
    frame.add_noise_from_obs(means, stddevs, mins, noise_type='chi2')

    # add RFI at random
    if np.random.choice([True, False], p=[rfi_prob, 1-rfi_prob]):
        add_rfi(frame)

    noise = np.copy(frame.get_data())

    # add signal to frame
    simulate_signal(frame, add_to_frame=True)
    signal = frame.get_data()

    # give true signal a target slope and slope of 0 to noise
    slope = utils.get_slope_from_driftRate(frame)

    return noise, signal, slope

def make_labels(num_samples, fchans, tchans, df, dt, min_freq, max_freq,
                    means, stddevs, mins, rfi_prob=0.15, num_cores=0):
    # pre-allocate arrays for ftdata and slopes
    ftdata = np.zeros([2 * num_samples, tchans, fchans])
    slopes = np.zeros(2 * num_samples)

    # make array of alternating training labels (faster than for loop)
    labels = np.zeros(2 * num_samples)
    labels[1::2] = 1

    # compactify all arguments into one tuple for multiprocessing
    simulation_args = parse_frame_args(fchans, tchans, df, dt, min_freq, max_freq,
                                        rfi_prob, means, stddevs, mins)

    # add pulses to frames only on odd-numbered samples
    print("Simulating signals in training backgrounds...")

    # create artificial frames and save them into training data
    if num_cores == 0: # make training data serially (single-core)
        for sample_number in tqdm.trange(num_samples):
            noise, signal, slope = make_artificial_frame(simulation_args)
            ftdata[2*sample_number, :, :] = noise
            ftdata[2*sample_number + 1, :, :] = signal
            slopes[2*sample_number + 1] = slope # set slope of noise to 0
    else:
        print(f"Running in parallel with {num_cores} cores")

        # duplicate simulation_args for parallel processes
        simulation_args_iterable = [simulation_args for i in np.arange(num_samples)]
        with mp.Pool(num_cores) as pool:
            result = pool.imap(make_artificial_frame, simulation_args_iterable)
            for sample_number in tqdm.trange(num_samples):
                noise, signal, slope = next(result)

                ftdata[2*sample_number, :, :] = noise
                ftdata[2*sample_number + 1, :, :] = signal
                slopes[2*sample_number + 1] = slope # set slope of noise to 0

    return ftdata, labels, slopes

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()

    ### SETIGEN FRAME PARAMETERS ###
    parser.add_argument('-p', '--path_to_files', nargs='+', type=str,
                        help='Regex pattern of matching .fil or .h5 names. Example: ./*0000.fil')

    parser.add_argument('-samp', '--num_samples', type=int, default=1000, help='Total number of samples to generate')
    parser.add_argument('-spf', '--samples_per_file', type=int, default=50,
                        help='Number of training samples to extract from each filterbank file')

    # control number of freq/time channels from each array.
    # Default value of None means entire time integration is used.
    parser.add_argument('-f', '--fchans', type=int, default=1024,
                        help='Number of frequency channels to extract for each training sample')
    parser.add_argument('-t', '--tchans', type=int, default=16,
                        help='Number of time bins to extract for each training sample. If None, use entire integration time')
    parser.add_argument('-fs', '--f_shift', type=float, default=None,
                        help='Number of frequency channels to extract for each training sample')
    parser.add_argument('-df', '--bandwidth', type=float, default=2.8, help='Frequency bandwidth; i.e. Hz per channel for simulated arrays.')
    parser.add_argument('-dt', '--sampling_time', type=float, default=18, help='Sampling time; number of seconds between bins for simulated arrays.')
    parser.add_argument('-fmin', '--min_freq', type=float, default=4e9, help='Minimum frequency (Hz) for simulated arrays.')
    parser.add_argument('-fmax', '--max_freq', type=float, default=8e9, help='Maximum frequency (Hz) for simulated arrays.')

    parser.add_argument('-max_time', '--max_sampling_time', type=int, default=600,
                        help='Max amount of time (seconds) to sample from files before duplicating')

    ### SIGNAL PARAMETERS (SNR, width, drift rate, etc.) ###
    # parameters for signal-to-noise ratio of FRB
    parser.add_argument('-snr', '--SNR_range', nargs=2, type=float, default=[10, 20], help='SNR range of signals, sampled from uniform distribution.')
    parser.add_argument('-drift', '--drift_rate', nargs=2, type=float, default=[-5, 5],
                            help='Min/max value for uniformly distributed drift rates (Hz/s)')
    parser.add_argument('--width', nargs=2, type=float, default=[10, 40], help='Min/max value for signal widths.')
    parser.add_argument('--rfi_prob', type=float, default=0.15, help='Probability of injecting RFI into simulated array.')

    # save training set
    parser.add_argument('--save_training_set', type=str, default=None,
                        help='Filename to save training set')

    # load in previously created training set
    parser.add_argument('-l', '--load_training', type=str, default=None,
                        help='Filename to load previously created training set (.npz file)')

    ### MODEL PARAMETERS ###
    # parameters for convolutional layers
    parser.add_argument('-conv', '--num_conv_layers', type=int, default=2, help='Number of convolutional layers to train with.')
    parser.add_argument('-filt', '--num_filters', type=int, default=32,
                        help='Number of filters in starting convolutional layer, doubles with every convolutional block')

    # parameters for dense layers
    parser.add_argument('-d1', '--n_dense1', type=int, default=256, help='Number of neurons in first dense layer')
    parser.add_argument('-d2', '--n_dense2', type=int, default=128, help='Number of neurons in second dense layer')

    parser.add_argument('-w', '--weight_signal', type=float, default=1.0,
                        help='Weight on true signal. Favor precision over recall if < 1 and vice-versa.')
    parser.add_argument('-split', '--train_val_split', type=float, default=0.5, help='Ratio to divide training and validation sets.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for model training')
    parser.add_argument('-e', '--epochs', type=int, default=32, help='Number of epochs to train with')

    # save the model, confusion matrix for last epoch, and validation set
    parser.add_argument('--previous_model', type=str, default=None,
                        help='Path to previous model, will be trained on new simulated data.')
    parser.add_argument('-s', '--save_model', dest='best_model_file', type=str,
                        default='./best_model.h5', help='Filename/path to save best model')
    parser.add_argument('-confusion', '--confusion_matrix', type=str,
                        default=None, help='Filename to store final confusion matrix')

    ### OPTIMIZATIONS ARGS ###
    parser.add_argument('--disable_numba', dest='enable_numba', action='store_false',
                        help='Disable numba speed optimizations')
    parser.add_argument('-cores', '--num_cores', type=int, default=0,
                        help='Number of cores to use for multiprocessing. Defaults to single-core.')

    args = parser.parse_args()

    path_to_files = args.path_to_files
    num_samples = args.num_samples
    samples_per_file = args.samples_per_file

    fchans = args.fchans
    tchans = args.tchans
    f_shift = args.f_shift

    max_sampling_time = args.max_sampling_time
    training_set_name = args.save_training_set

    # Args for model parameters
    prev_training_set = args.load_training
    saved_model_name = args.best_model_file
    previous_model = args.previous_model

    script_start_time = time() # time how long script takes to run

    if prev_training_set: # override -p argument if loading in training set
        print(f"Loading in previously created training set: {prev_training_set}\n")
        training_params = np.load(prev_training_set, allow_pickle=True)
        means, stddevs, mins = training_params['means'], training_params['stddevs'], training_params['mins']
    else:
        if path_to_files is None:
            raise ValueError("-p (path_to_files) must be specified when creating training set from scratch")

        print("Creating training set from scratch...\n")
        means, stddevs, mins = generate_dataset.main(path_to_files, fchans, tchans, f_shift,
                                            samples_per_file, num_samples, max_sampling_time)

        if training_set_name:
            # save final array to disk
            print("Saving training set to " + training_set_name)
            np.savez(training_set_name, means=means, stddevs=stddevs, mins=mins)

    print(f'Number of frequency channels per sample: {fchans}')
    print(f'Number of time bins per sample: {tchans}')
    print('\n')

    ftdata, labels, slopes = make_labels(num_samples=num_samples, fchans=fchans, tchans=tchans, df=args.bandwidth, dt=args.sampling_time,
                                            min_freq=args.min_freq, max_freq=args.max_freq, rfi_prob=args.rfi_prob,
                                            means=means, stddevs=stddevs, mins=mins, num_cores=args.num_cores)

    start_time = time()
    print('\nScaling arrays...')
    if args.enable_numba: # use numba-accelerated functions
        utils.scale_data_numba(ftdata)
    else:
        utils.scale_data(ftdata)
    print(f"Done scaling in {time() - start_time: .2f} seconds!\n")


    # split data into training and validation sets
    start_time = time()
    print('Splitting data into training and validation sets...')
    train_ftdata, val_ftdata, train_labels, val_labels, train_slopes, val_slopes = train_test_split(ftdata, labels, slopes, train_size=args.train_val_split)
    print(f"Split data in {time() - start_time: .2f} seconds!\n")

    ftdata = labels = slopes = None # free memory by deleting potentially huge arrays

    # disable file locking to save neural network models
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    print("Constructing CNN with given input parameters...")
    model = construct_model(num_conv_layers=args.num_conv_layers, num_filters=args.num_filters,
                                n_dense1=args.n_dense1, n_dense2=args.n_dense2,
                                saved_model_name=saved_model_name, previous_model=previous_model)
    print(model.summary())

    # add channel dimension for Keras tensors (1 channel)
    train_ftdata = train_ftdata[..., None]
    val_ftdata = val_ftdata[..., None]

    fit_model(model, train_ftdata, train_labels, val_ftdata, val_labels,
            train_slopes, val_slopes, saved_model_name=saved_model_name,
            weight_signal=args.weight_signal, batch_size=args.batch_size, epochs=args.epochs)

    print(f"\nTraining and validating on {num_samples} samples took {(time() - start_time) / 60:.2f} minutes")

    # load the best model saved to generate confusion matrix
    print("Evaluating on validation set to generate confusion matrix...")
    model = load_model(saved_model_name, compile=False)
    pred_probs = model.predict(val_ftdata, verbose=1)[0].flatten()

    utils.plot_confusion_matrix(val_ftdata, val_labels, pred_probs, args.confusion_matrix)

    print(f"\nGenerating training set + training model + evaluating predictions: {(time() - script_start_time) / 60:.2f} minutes")j