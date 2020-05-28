import numpy as np

# simulate pulses
import setigen as stg

from time import time
import os, sys, glob, tqdm
import argparse
import copy

"""
@source Bryan Brzycki (https://github.com/bbrzycki)
"""

def extract_frames(fname, training_frames, fchans=1024, tchans=None,
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

    # grab background samples from observation and simulate pulse
    for sample_number in tqdm.trange(samples_per_file):
        if len(training_frames) < total_samples:
            try: # make setigen frame if next iteration is possible
                wt_obs = next(waterfall_itr)
                frame = stg.Frame(waterfall=wt_obs)
            except StopIteration: # break if we reach end of file
                print("End of file, reduce f_shift or samples_per_file if this keeps happening.")
                break

            training_frames.append(frame) # add frame to list of frames
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
    start_time = time()
    training_frames = []

    # sample files that follow regex pattern of path_to_files
    # so path_to_files = './*0000.fil' would sample all files
    # in current directory that end in 0000.fil
    if isinstance(path_to_files, list):
        files = path_to_files
    elif isinstance(path_to_files, str):
        files = glob.glob(path_to_files)
    else:
        raise ValueError(f"path_to_files should be list or str type, not {type(path_to_files)}.")

    # if given None, sampling will continue until out of files or total_samples is achieved
    if max_sampling_time is None:
        max_sampling_time = np.inf

    print(f"Total number of files to possibly sample from: {len(files)}")
    print(f"Total number of samples to get: {total_samples}")

    if not files:
        raise ValueError(f"No files found with path {path_to_files}")

    # choose number of files to sample from based on
    # user-inputted sample size or initial number of files
    num_files = len(files)

    print(f"Randomly grabbing {samples_per_file} samples from each file")
    random_files = np.random.choice(files, size=num_files, replace=False)

    i = 0
    loop_start = time()
    while len(training_frames) < total_samples:
        elapsed_time = time() - loop_start
        print(f"Elapsed time: {np.round(elapsed_time / 60, 2)} minutes")

        # end scanning if we looked through all files or takes too long
        if i >= len(random_files) or elapsed_time >= max_sampling_time:
            if i >= len(random_files):
                print("\nOut of files to sample without replacement. Either increase samples_per_file or find more files to sample from.")
            if elapsed_time >= max_sampling_time:
                print("\nExceeded max sampling time.")

            # augment training set with duplicates
            # beware! if too many duplicates, model will not generalize well
            print("Duplicating samples...")
            duplicate_samples(training_frames, total_samples)
            break

        # pick a random filterbank file from directory
        rand_filename = random_files[i]
        print(f"\nSampling file ({i}/{len(files)}): " + str(rand_filename))

        # get information and append to growing list of samples
        extract_frames(fname=rand_filename, training_frames=training_frames,
                        fchans=fchans, tchans=tchans, f_shift=f_shift,
                        samples_per_file=samples_per_file, total_samples=total_samples)

        i += 1
        print("Number of samples after scan: " + str(len(training_frames)))

    print(f"\nUnique number of files used: {i}")

    training_frames = np.array(training_frames)

    print(f"Training set creation time: {np.round((time() - start_time) / 60, 4)} min")
    return training_frames

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
    parser.add_argument('-t', '--tchans', type=int, default=None,
                        help='Number of time bins to extract for each training sample. If None, use entire integration time')
    parser.add_argument('-fs', '--f_shift', type=float, default=None,
                        help='Number of frequency channels from start of current frame to begin successive frame. If None, default to no overlap, i.e. f_shift=fchans).')

    parser.add_argument('-max_time', '--max_sampling_time', type=int, default=600,
                        help='Max amount of time (seconds) to sample from files before duplicating')

    # save training set
    parser.add_argument('-s', '--save_name', type=str, default='training_set.npy',
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

    training_frames = main(path_to_files, fchans, tchans, f_shift,
                            samples_per_file, total_samples, max_sampling_time)

    # save final array to disk
    print("Saving data to " + save_name)
    np.save(save_name, training_frames)