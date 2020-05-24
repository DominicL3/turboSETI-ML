import numpy as np

# simulate pulses
import setigen as stg

from time import time
import os, sys, glob, tqdm
import copy

import numba # speed up Numpy

"""
@source Bryan Brzycki (https://github.com/bbrzycki)
"""

def extract_frames(fname, fchans, tchans, f_shift=None, training_frames, training_labels,
                    total_samples, samples_per_file=50):
    """
    Given a filename, takes time samples from filterbank file
    and converts them into Spectra objects, which will then be
    randomly dedispersed and used to inject FRBs into.

    Returns:
        training_frames : numpy.ndarray
            Array of Setigen frames.
        training_labels : numpy.ndarray
            Array of binary training labels, alternating between 0 and 1.
    """

    # grab samples_per_file number of samples from filterbank file
    waterfall_itr = stg.split_waterfall_generator(fname, fchans, tchans, f_shift=f_shift)

    # grab background samples from observation and simulate pulse
    for sample_number in tqdm.trange(samples_per_file):
        if len(training_frames) < total_samples:
            try: # make setigen frame if next iteration is possible
                wt_obs = next(waterfall_itr)
                frame = stg.Frame(waterfall=wt_obs)
            except StopIteration: # break if we reach end of file
                break

            # simulate a pulse for every sample
            simulate_pulse(frame, training_frames, training_labels)
        else:
            break

def simulate_pulse(frame, training_frames, training_labels):
    """Generate dataset, taken from setigen docs (advanced topics)."""

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

    # add blank observation with label 0 to training set
    training_frames.append(copy.deepcopy(frame))
    training_labels.append(0)

    # add signal to background
    signal = frame.add_constant_signal(f_start=frame.get_frequency(start_index),
                                drift_rate=drift_rate,
                                level=frame.get_intensity(snr=random_SNR),
                                width=width,
                                f_profile_type=f_profile_type)


    # append frame with signal to training set
    training_frames.append(copy.deepcopy(frame))
    training_labels.append(1)

def duplicate_samples(current_samples, total_samples):
    """
    Chooose random samples and copy them so that len(current_samples) == total_samples.
    This is done if there isn't enough data after collecting samples through all files.

    Returns:
        current_samples : list
        Modified list of arrays, with duplicates at the end.
    """
    duplicates = np.random.choice(current_samples, size=total_samples - len(current_samples))
    current_samples.extend(duplicates)

    return current_samples

@numba.njit(parallel=True)
def duplicate_samples_numba(current_samples, total_samples):
    """
    Chooose random samples and copy them so that len(current_samples) == total_samples.
    This is done if there isn't enough data after collecting samples through all files.

    Returns:
        current_samples : list
        Modified list of arrays, with duplicates at the end.
    """
    duplicates = np.random.choice(current_samples, size=total_samples - len(current_samples))
    current_samples.extend(duplicates)

    return current_samples

def main(path_to_files, fchans=1024, tchans=None, f_shift=None,
            samples_per_file=50, total_samples=1000,
            max_sampling_time=600, enable_numba=True):
    start_time = time()
    training_frames, training_labels = [], []

    # sample files that follow regex pattern of path_to_files
    # so path_to_files = './*0000.fil' would sample all files
    # in current directory that end in 0000.fil
    files = glob.glob(path_to_files)

    print("Total number of files to possibly sample from: %d" % len(files))

    if not files:
        raise ValueError(f"No files found with path {path_to_files}")

    # choose number of files to sample from based on
    # user-inputted sample size or initial number of files
    num_files = len(files)

    print(f"{num_files} total files, randomly grabbing {samples_per_file} samples each")
    random_files = np.random.choice(files, size=num_files, replace=False)

    i = 0
    loop_start = time()
    while len(training_samples) < total_samples:
        elapsed_time = time() - loop_start
        print(f"Elapsed time: {np.round(elapsed_time / 60, 2)} minutes")

        # end scanning if we looked through all files or takes too long
        if i >= len(random_files) or elapsed_time >= max_sampling_time:
            print("\nTaking too long. Duplicating samples...")
            if enable_numba:
                duplicate_samples_numba(training_samples, total_samples) # copy samples
            else:
                duplicate_samples(training_samples, total_samples) # copy samples
            break

        # pick a random filterbank file from directory
        rand_filename = random_files[i]
        print("\nSampling file: " + str(rand_filename))

        # get information and append to growing list of samples
        extract_frames(fname=rand_filename, fchans=fchans, tchans=tchans, f_shift=f_shift,
                        training_frames=training_frames, training_labels=training_labels,
                        total_samples=total_samples, samples_per_file=samples_per_file)

        i += 1
        print("Number of samples after scan: " + str(len(training_samples)))

    print(f"Unique number of files after random sampling: {len(np.unique(random_files))}")

    training_frames = np.array(training_frames)
    training_labels = np.array(training_labels)

    print(f"Training set creation time: {np.round((time() - start_time) / 60, 4)} min")
    return training_frames, training_labels

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_files', type=str, required=True,
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
    parser.add_argument('--disable_numba', dest='enable_numba', action='store_false',
                        help='Turn off Numba speed optimizations')

    # save training set
    parser.add_argument('-s', '--save_name', type=str, default='training_set.npz',
                        help='Filename to save training set')

    args = parser.parse_args()

    path_to_files = args.path_to_files
    total_samples = args.total_samples
    samples_per_file = args.samples_per_file

    fchans = args.fchans
    tchans = args.tchans
    f_shift = args.f_shift

    max_sampling_time = args.max_sampling_time
    enable_numba = args.enable_numba
    save_name = args.save_name

    training_frames, training_labels = main(path_to_files, fchans, tchans, f_shift,
                                            samples_per_file, total_samples,
                                            max_sampling_time, enable_numba)

    # save final array to disk
    print("Saving data to " + save_name)
    np.savez(save_name, frames=training_frames, labels=training_labels)