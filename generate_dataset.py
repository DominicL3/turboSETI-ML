import numpy as np

# simulate pulses
import setigen as stg
from astropy.stats import sigma_clip

from time import time
import os, sys, glob, tqdm
import argparse
import copy

"""
@source Bryan Brzycki (https://github.com/bbrzycki)
"""

def get_data_stats(wt_obs):
    clipped_data = sigma_clip(stg.waterfall_utils.get_data(wt_obs),
                                  sigma=3,
                                  maxiters=5,
                                  masked=False)
    x_mean = np.mean(clipped_data)
    x_std = np.std(clipped_data)
    x_min = np.min(clipped_data)

    return x_mean, x_std, x_min

def extract_frames(fname, means, stddevs, mins, fchans=1024, tchans=None,
                    f_shift=None, samples_per_file=50, total_samples=1000):
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

    # grab background samples from observation
    for sample_number in tqdm.trange(samples_per_file):
        if len(means) < total_samples:
            try: # make setigen frame if next iteration is possible
                wt_obs = next(waterfall_itr)
                x_mean, x_std, x_min = get_data_stats(wt_obs)
            except StopIteration: # break if we reach end of file
                print("End of file, reduce f_shift or samples_per_file if this keeps happening.")
                break

            means.append(x_mean)
            stddevs.append(x_std)
            mins.append(x_min)

        else:
            break

def duplicate_samples(current_samples, total_samples):
    """
    Chooose random samples and copy them so that len(current_samples) == total_samples.
    This is done if there isn't enough data after collecting samples through all files.

    Returns:
        current_samples : list
        Modified list of arrays, with duplicates at the end.
    """
    duplicates = np.random.choice(current_samples, size=total_samples - len(current_samples))
    current_samples.extend([copy.deepcopy(frame) for frame in duplicates])

    return current_samples


def main(path_to_files, fchans=1024, tchans=None, f_shift=None,
            samples_per_file=50, total_samples=1000, max_sampling_time=600):
    means, stddevs, mins = [], [], []

    # sample files that follow regex pattern of path_to_files
    # so path_to_files = './*0000.fil' would sample all files
    # in current directory that end in 0000.fil
    if isinstance(path_to_files, list):
        files = path_to_files
    elif isinstance(path_to_files, str):
        files = glob.glob(path_to_files)
    else:
        raise ValueError(f"path_to_files should be list or str type, not {type(path_to_files)}.")

    if not files:
        raise ValueError(f"No files found with path {path_to_files}")

    # if given None, sampling will continue until out of files or total_samples is achieved
    if max_sampling_time == 0:
        max_sampling_time = np.inf

    print(f"Total number of files to possibly sample from: {len(files)}")
    print(f"Total number of samples to get: {total_samples}")

    print(f"Randomly grabbing {samples_per_file} samples from each file")
    np.random.shuffle(files) # randomize order of files to sample from

    i = 0
    loop_start = time()
    while len(means) < total_samples:
        elapsed_time = time() - loop_start
        print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")

        # end scanning if we looked through all files or takes too long
        if i >= len(files) or elapsed_time >= max_sampling_time:
            if i >= len(files):
                print("\nOut of files to sample without replacement. Either increase samples_per_file or find more files to sample from.")
            if elapsed_time >= max_sampling_time:
                print("\nExceeded max sampling time.")
            break

        # pick a random filterbank file from directory
        rand_filename = files[i]
        print(f"\nSampling file ({i}/{len(files)}): " + str(rand_filename))

        # get information and append to growing list of samples
        extract_frames(fname=rand_filename, means=means, stddevs=stddevs, mins=mins,
                        fchans=fchans, tchans=tchans, f_shift=f_shift,
                        samples_per_file=samples_per_file, total_samples=total_samples)

        i += 1
        print("Number of samples after scan: " + str(len(means)))

    print(f"\nUnique number of files used: {i}")

    means, stddevs, mins = np.array(means), np.array(stddevs), np.array(mins)
    return means, stddevs, mins

if __name__ == "__main__":
    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_files', nargs='+', type=str, help='Regex pattern of matching .fil or .h5 names. Example: ./*0000.fil')

    parser.add_argument('-total', '--total_samples', type=int, default=1000, help='Total number of samples to generate')
    parser.add_argument('-spf', '--samples_per_file', type=int, default=50,
                        help='Number of training samples to extract from each filterbank file')

    # control number of freq/time channels from each array.
    # Default value of None means all time channels are used.
    parser.add_argument('-f', '--fchans', type=int, default=1024,
                        help='Number of frequency channels to extract for each training sample')
    parser.add_argument('-t', '--tchans', type=int, default=16,
                        help='Number of time bins to extract for each training sample. If None, use entire integration time')
    parser.add_argument('-fs', '--f_shift', type=float, default=None,
                        help='Number of frequency channels from start of current frame to begin successive frame. If None, default to no overlap, i.e. f_shift=fchans).')

    parser.add_argument('-max_time', '--max_sampling_time', type=float, default=0,
                        help='Max amount of time (seconds) to sample from files before duplicating. If 0, there is no limit on the sampling time.')

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
    save_name = args.save_name

    script_start_time = time()

    means, stddevs, mins = main(path_to_files, fchans, tchans, f_shift,
                                samples_per_file, total_samples, max_sampling_time)

    print(f"Ended with {len(means)} out of {total_samples} samples.")

    # save final array to disk
    print("Saving data to " + save_name)
    np.savez(save_name, means=means, stddevs=stddevs, mins=mins)

    print(f"\nTraining set creation time: {(time() - script_start_time) / 60:.2f} min")