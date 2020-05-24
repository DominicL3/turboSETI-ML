import numpy as np
import tqdm

"""
Helper functions for training neural network, including
data preprocessing and computing training results.

@source Liam Connor (https://github.com/liamconnor/single_pulse_ml)
"""

def split(array, bins_per_array):
    """
    Splits long 2D array into 3D array of multiple 2D arrays,
    such that each has bins_per_array time bins. Drops the
    last chunk if it has fewer than bins_per_array bins.

    Returns:
        split_array : numpy.ndarray
            Array after splitting.
    """

    total_bins = array.shape[1]

    split_array = np.zeros((int(np.ceil(total_bins/bins_per_array)), array.shape[0], bins_per_array),
                          dtype=array.dtype)

    for i in numba.prange(len(split_array)):
        split_array[i] = array[:, i * bins_per_array:(i+1) * bins_per_array]

    if total_bins % bins_per_array != 0: # fix when unevenly split
        # last array currently only filled partially
        # set last array in split_array to larger chunk from the end
        split_array[-1] = array[:, -bins_per_array:]

    return split_array

# chop_off WITH Numba support
@numba.njit(parallel=True)
def split_numba(array, bins_per_array):
    """
    Does the same thing as split() but uses Numba for
    increased speed performance.

    Splits long 2D array into 3D array of multiple 2D arrays,
    such that each has bins_per_array time bins. Drops the last chunk if it
    has fewer than bins_per_array bins.

    Returns:
        split_array : numpy.ndarray
            Array after splitting.
    """
    total_bins = array.shape[1]

    split_array = np.zeros((int(np.ceil(total_bins/bins_per_array)), array.shape[0], bins_per_array),
                          dtype=array.dtype)

    for i in numba.prange(len(split_array)):
        split_array[i] = array[:, i * bins_per_array:(i+1) * bins_per_array]

    if total_bins % bins_per_array != 0: # fix when unevenly split
        # last array currently only filled partially
        # set last array in split_array to larger chunk from the end
        split_array[-1] = array[:, -bins_per_array:]

    return split_array

# reshaping version of split()
# used when total_bins % bins_per_array = 0
def split_reshape(array, bins_per_array):
    """
    Same function as split() and split_numba(), but takes advantage
    when bins_per_array divides evenly into total_bins to reshape,
    since reshaping is much faster than np.split().

    Returns:
        split_array : numpy.ndarray
            Array after splitting.
    """
    num_blocks = array.shape[1] / bins_per_array
    split_array = array.reshape(int(num_blocks), array.shape[0], bins_per_array)

    return split_array

def scale_data(ftdata):
    """Subtract each channel in 3D array by its median and
    divide each array by its global standard deviation. Perform
    this standardization in chunks to avoid a memory overload."""

    N = 10000
    for i in tqdm.trange(int(np.ceil(len(ftdata)/float(N)))):
        ftdata_chunk = ftdata[i*N:(i + 1) * N]
        medians = np.median(ftdata_chunk, axis=-1)[:, :, np.newaxis]
        stddev = np.std(ftdata_chunk.reshape(len(ftdata_chunk), -1), axis=-1)[:, np.newaxis, np.newaxis]

        scaled_ftdata = (ftdata_chunk - medians) / stddev
        ftdata[i*N:(i + 1) * N] = scaled_ftdata

# rescaling with Numba
@numba.njit(parallel=True)
def scale_data_numba(ftdata):
    """Numba-accelerated version of scale_data() for vast
    improvements in speed, especially for large arrays.

    Subtract each channel in 3D array by its median and
    divide each array by its global standard deviation.
    """
    for chunk_idx in numba.prange(len(ftdata)):
        rescaled_chunk = ftdata[chunk_idx] # iterate over every 2D array
        stddev = np.std(rescaled_chunk)
        for row_idx in numba.prange(len(rescaled_chunk)): # subtract median from each row
            rescaled_chunk[row_idx, :] -= np.median(rescaled_chunk[row_idx, :])
        rescaled_chunk[:, :] /= stddev # divide every 2D array by its stddev

        ftdata[chunk_idx] = rescaled_chunk

def get_classification_results(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted
    label (y_pred) for a binary classifier, and return
    true_positives, false_positives, true_negatives, false_negatives
    """
    true_positives = np.where((y_true == 1) & (y_pred >= 0.5))[0]
    false_positives = np.where((y_true == 0) & (y_pred >= 0.5))[0]
    true_negatives = np.where((y_true == 0) & (y_pred < 0.5))[0]
    false_negatives = np.where((y_true == 1) & (y_pred < 0.5))[0]

    return true_positives, false_positives, true_negatives, false_negatives

def print_metric(y_true, y_pred):
    """ Take true labels (y_true) and model-predicted
    label (y_pred) for a binary classifier
    and print a confusion matrix, metrics,
    return accuracy, precision, recall, fscore
    """

    def confusion_mat(y_true, y_pred):
        """ Generate a confusion matrix for a binary classifier
        based on true labels (y_true) and model-predicted label (y_pred)

        returns np.array([[TP, FP],[FN, TN]])
        """
        TP, FP, TN, FN = get_classification_results(y_true, y_pred)

        NTP = len(TP)
        NFP = len(FP)
        NTN = len(TN)
        NFN = len(FN)

        conf_mat = np.array([[NTP, NFP], [NFN, NTN]])
        return conf_mat

    conf_mat = confusion_mat(y_true, y_pred)

    NTP, NFP, NTN, NFN = conf_mat[0, 0], conf_mat[0, 1], conf_mat[1, 1], conf_mat[1, 0]

    print("Confusion matrix:")

    print('\n'.join([''.join(['{:8}'.format(item) for item in row])
                     for row in conf_mat]))

    accuracy = (NTP + NTN) / conf_mat.sum()
    precision = NTP / (NTP + NFP)
    recall = NTP / (NTP + NFN)
    fscore = 2 * precision * recall / (precision + recall)

    print("accuracy: %f" % accuracy)
    print("precision: %f" % precision)
    print("recall: %f" % recall)
    print("fscore: %f" % fscore)

    return accuracy, precision, recall, fscore, conf_mat