import numpy as np

from tqdm import tqdm
from time import time
import os, argparse

# neural net imports
from tensorflow.keras.models import load_model
from model import construct_conv2d, fit_model

import generate_dataset, utils # make training set

"""
Train a Keras model to do binary classification on simulated pulses
vs. background RFI and save the best model from training. Exit automatically
if validation loss doesn't improve after a certain number of epochs.

Takes in a .fil file to use as background RFI and uses setigen to simulate
narrowband signals with randomly generated signal properties.

@source Liam Connor (https://github.com/liamconnor/single_pulse_ml)
@source Bryan Brzycki (https://github.com/bbrzycki/setigen)
"""

def simulate_pulse(frame, add_to_frame=True):
    """Generate dataset, taken from setigen docs (advanced topics)."""
    fchans = frame.fchans

    # let true pulse begin in middle 50% of array and randomize drift rate
    start_index = np.random.randint(0.25 * fchans, 0.75 * fchans)
    drift_rate = np.random.uniform(-5, 5)

    # sample SNR and frequency profile randomly
    random_SNR = 8 + np.random.lognormal(mean=1.0, sigma=1.0)
    width = np.random.uniform(10, 40)
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

def make_labels(training_frames):
    # pre-allocate training data array using first frame
    f0 = training_frames[0]
    fchans, tchans = f0.fchans, f0.tchans
    print(f'Number of frequency channels per sample: {fchans}')
    print(f'Number of time bins per sample: {tchans}')
    print('\n')

    ftdata = np.zeros([len(training_frames), tchans, fchans], dtype=f0.get_data().dtype)

    # add pulses to frames only on odd-numbered samples
    print("Simulating pulses in training backgrounds")
    for sample_number, frame in enumerate(tqdm(training_frames)):
        if sample_number % 2 == 0:
            # add blank observation to training set
            simulate_pulse(frame, add_to_frame=False)
        else:
            # add signal to frame
            simulate_pulse(frame, add_to_frame=True)

        ftdata[sample_number, :, :] = frame.get_data()

    # make array of alternating training labels
    labels = np.zeros(len(training_frames))
    labels[1::2] = 1

    return ftdata, labels

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()

    ### SIMULATION PARAMETERS ###
    parser.add_argument('-p', '--path_to_files', nargs='+', type=str,
                        help='Regex pattern of matching .fil or .h5 names. Example: ./*0000.fil')

    parser.add_argument('-total', '--total_samples', type=int, default=1000, help='Total number of samples to generate')
    parser.add_argument('-spf', '--samples_per_file', type=int, default=50,
                        help='Number of training samples to extract from each filterbank file')

    # control number of freq/time channels from each array.
    # Default value of None means all time channels are used.
    parser.add_argument('-f', '--fchans', type=int, default=1024,
                        help='Number of frequency channels to extract for each training sample')
    parser.add_argument('-t', '--tchans', type=int, default=None,
                        help='Number of time bins to extract for each training sample. If None, use entire integration time')
    parser.add_argument('--f_shift', type=float, default=None,
                        help='Number of frequency channels to extract for each training sample')

    parser.add_argument('-max_time', '--max_sampling_time', type=int, default=600,
                        help='Max amount of time (seconds) to sample from files before duplicating')

    # save training set
    parser.add_argument('--save_training_set', type=str, default=None,
                        help='Filename to save training set')

    # load in previously created training set
    parser.add_argument('-l', '--load_training', type=str, default=None,
                        help='Filename to load previously created training set (.npy file)')

    ### MODEL PARAMETERS ###
    # parameters for convolutional layers
    parser.add_argument('-conv', '--num_conv_layers', type=int, default=3, help='Number of convolutional layers to train with.')
    parser.add_argument('-filt', '--num_filters', type=int, default=32,
                        help='Number of filters in starting convolutional layer, doubles with every convolutional block')

    # parameters for dense layers
    parser.add_argument('-d1', '--n_dense1', type=int, default=256, help='Number of neurons in first dense layer')
    parser.add_argument('-d2', '--n_dense2', type=int, default=128, help='Number of neurons in second dense layer')

    # parameters for signal-to-noise ratio of FRB
    parser.add_argument('--SNRmin', type=float, default=8.0, help='Minimum SNR for FRB signal')
    parser.add_argument('--SNR_sigma', type=float, default=1.0, help='Standard deviation of SNR from log-normal distribution')
    parser.add_argument('--SNRmax', type=float, default=30.0, help='Maximum SNR of FRB signal')

    parser.add_argument('-w', '--weight_signal', type=float, default=1.0,
                        help='Class weight of true signal, used to favor false positives (< 1) or false negatives (> 1)')

    parser.add_argument('-split_frac', '--train_val_split', type=float, default=0.5, help='Ratio to divide training and validation sets.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for model training')
    parser.add_argument('-e', '--epochs', type=int, default=32, help='Number of epochs to train with')

    # save the model, confusion matrix for last epoch, and validation set
    parser.add_argument('--previous_model', type=str, default=None,
                        help='Path to previous model, will be trained on new simulated data.')
    parser.add_argument('-s', '--save_model', dest='best_model_file', type=str,
                        default='./best_model.h5', help='Filename/path to save best model')
    parser.add_argument('-confusion', '--confusion_matrix', type=str,
                        default=None, help='Filename to store final confusion matrix')
    parser.add_argument('--disable_numba', dest='enable_numba', action='store_false',
                        help='Disable numba speed optimizations')

    args = parser.parse_args()

    path_to_files = args.path_to_files
    total_samples = args.total_samples
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

    if prev_training_set: # override -p argument if loading in training set
        print(f"Loading in previously created training set: {prev_training_set}\n")
        training_frames = np.load(prev_training_set, allow_pickle=True)
    else:
        if path_to_files is None:
            raise ValueError("-p (path_to_files) must be specified when creating training set from scratch")

        print("Creating training set from scratch...\n")
        training_frames = generate_dataset.main(path_to_files, fchans, tchans, f_shift,
                                            samples_per_file, total_samples, max_sampling_time)

        if training_set_name:
            # save final array to disk
            print("Saving training set to " + training_set_name)
            np.save(training_set_name, training_frames)

    ftdata, labels = make_labels(training_frames)

    if args.enable_numba: # use numba-accelerated functions
        start_time = time()
        print('Scaling arrays...')
        utils.scale_data_numba(ftdata)
        print(f"Done scaling in {np.round((time() - start_time), 2)} seconds!\n")

        start_time = time()
        print('Splitting data into training and validation sets')
        train_ftdata, train_labels, val_ftdata, val_labels = utils.train_val_split_numba(ftdata, labels, args.train_val_split)
        print(f"Split data in {np.round((time() - start_time), 2)} seconds!\n")
    else:
        print('Scaling arrays...')
        utils.scale_data(ftdata)
        print(f"Done scaling in {np.round((time() - start_time), 2)} seconds!\n")

        start_time = time()
        print('Splitting data into training and validation sets')
        train_ftdata, train_labels, val_ftdata, val_labels = utils.train_val_split(ftdata, labels, args.train_val_split)
        print(f"Split data in {np.round((time() - start_time), 2)} seconds!\n")

    ftdata = None; del ftdata # free memory by deleting potentially huge array

    # disable file locking to save NN models
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    print("Constructing CNN with input parameters")
    model = construct_conv2d(num_conv_layers=args.num_conv_layers, num_filters=args.num_filters,
                                n_dense1=args.n_dense1, n_dense2=args.n_dense2,
                                saved_model_name=saved_model_name, previous_model=previous_model)
    print(model.summary())

    start_time = time() # training time

    # add channel dimension for Keras tensors (1 channel)
    train_ftdata = train_ftdata[..., None]
    val_ftdata = val_ftdata[..., None]

    fit_model(model, train_ftdata, train_labels, val_ftdata, val_labels,
                saved_model_name=saved_model_name, weight_signal=args.weight_signal,
                batch_size=args.batch_size, epochs=args.epochs)

    print(f"\nTraining and validating on {len(labels)} samples took {np.round((time() - start_time) / 60, 2)} minutes")

    # load the best model saved to generate confusion matrix
    print("Evaluating on validation set to generate confusion matrix...")
    model = load_model(saved_model_name, compile=True)
    pred_probs = model.predict(val_ftdata, verbose=1)[:, 0]

    utils.plot_confusion_matrix(val_ftdata, val_labels, pred_probs,
                                args.confusion_matrix, enable_numba=args.enable_numba)