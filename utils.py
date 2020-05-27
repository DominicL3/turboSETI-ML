import numpy as np
import matplotlib.pyplot as plt
import numba # speed up NumPy
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
    """Subtract each frequency channel in 3D array by its median and
    divide each array by its global standard deviation. Perform
    this standardization in chunks to avoid a memory overload."""

    N = 10000
    for i in tqdm.trange(int(np.ceil(len(ftdata)/float(N)))):
        ftdata_chunk = ftdata[i*N:(i + 1) * N]
        medians = np.median(ftdata_chunk, axis=1)[:, :, np.newaxis]
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
    num_rows = ftdata.shape[1]
    for chunk_idx in numba.prange(len(ftdata)):
        rescaled_chunk = ftdata[chunk_idx] # iterate over every 2D array
        stddev = np.std(rescaled_chunk)
        for row_idx in numba.prange(num_rows): # subtract median from each row
            rescaled_chunk[row_idx, :] -= np.median(rescaled_chunk[row_idx, :])
        rescaled_chunk[:, :] /= stddev # divide every 2D array by its stddev

        ftdata[chunk_idx] = rescaled_chunk

def train_val_split(ftdata, labels, split_fraction):
    """Split ftdata and labels into training and validation sets."""
    NTRAIN = int(len(ftdata) * split_fraction) # split_fraction defines what proportion is training set

    ind = np.arange(len(ftdata))
    np.random.shuffle(ind)

    # split indices into training and evaluation set
    ind_train = ind[:NTRAIN]
    ind_val = ind[NTRAIN:]

    # split examples into training and test set based on randomized indices
    train_ftdata, val_ftdata = ftdata[ind_train], ftdata[ind_val]
    train_labels, val_labels = labels[ind_train], labels[ind_val]

    return train_ftdata, train_labels, val_ftdata, val_labels

@numba.njit(parallel=True)
def train_val_split_numba(ftdata, labels, split_fraction):
    """Literally the same function as train_val_split, but
    runs with Numba for speed optimizations."""

    NTRAIN = int(len(ftdata) * split_fraction) # split_fraction defines what proportion is training set

    ind = np.arange(len(ftdata))
    np.random.shuffle(ind)

    # split indices into training and evaluation set
    ind_train = ind[:NTRAIN]
    ind_val = ind[NTRAIN:]

    # split examples into training and test set based on randomized indices
    train_ftdata, val_ftdata = ftdata[ind_train], ftdata[ind_val]
    train_labels, val_labels = labels[ind_train], labels[ind_val]

    return train_ftdata, train_labels, val_ftdata, val_labels

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

@numba.njit(parallel=True)
def get_classification_results_numba(y_true, y_pred):
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

def plot_confusion_matrix(val_ftdata, val_labels, pred_probs, confusion_matrix_name, enable_numba=True):
    pred_labels = np.round(pred_probs)
    # print out scores of various metrics
    accuracy, precision, recall, fscore, conf_mat = print_metric(val_labels, pred_labels)
    ftdata_shape = val_ftdata[0, :, :, 0].shape

    if enable_numba:
        TP, FP, TN, FN = get_classification_results_numba(val_labels, pred_labels)
    else:
        TP, FP, TN, FN = get_classification_results(val_labels, pred_labels)

    # get lowest confidence selection for each category
    if TP.size:
        TPind = TP[np.argmin(pred_probs[TP])]  # Min probability True positive candidate
        TPdata = val_ftdata[..., 0][TPind]
    else:
        TPdata = np.zeros(ftdata_shape)

    if FP.size:
        FPind = FP[np.argmax(pred_probs[FP])]  # Max probability False positive candidate
        FPdata = val_ftdata[..., 0][FPind]
    else:
        FPdata = np.zeros(ftdata_shape)

    if FN.size:
        FNind = FN[np.argmax(pred_probs[FN])]  # Max probability False negative candidate
        FNdata = val_ftdata[..., 0][FNind]
    else:
        FNdata = np.zeros(ftdata_shape)

    if TN.size:
        TNind = TN[np.argmin(pred_probs[TN])]  # Min probability True negative candidate
        TNdata = val_ftdata[..., 0][TNind]
    else:
        TNdata = np.zeros(ftdata_shape)

    # plot the confusion matrix and display
    plt.ion()

    conf_mat = conf_mat.flatten()
    names = ['TP', 'FP', 'FN', 'TN']
    confusion_data = [TPdata, FPdata, FNdata, TNdata]

    fig_confusion, ax_confusion = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    for num_samples, name, data, ax in zip(conf_mat, names, confusion_data, ax_confusion.flatten()):
        ax.imshow(data, aspect='auto')
        ax.set_title(f'{name}: {num_samples}')

    fig_confusion.tight_layout()
    fig_confusion.show()

    # save plot to disk
    if confusion_matrix_name is not None:
        print("Saving confusion matrix to {}".format(confusion_matrix_name))
        fig_confusion.savefig(confusion_matrix_name, dpi=100)