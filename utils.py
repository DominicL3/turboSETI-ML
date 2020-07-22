import numpy as np
import matplotlib.pyplot as plt
import numba # speed up NumPy

from tqdm import tqdm, trange

from skimage import filters, morphology, measure, transform # classically detect line in image
from sklearn.linear_model import LinearRegression
from scipy.stats import theilslopes

import warnings

"""
Helper functions for training neural network, including
data preprocessing and computing training results.

@source Liam Connor (https://github.com/liamconnor/single_pulse_ml)
"""

def split(array, bins_per_array, bin_shift):
    """
    Splits long 2D array into 3D array of multiple 2D arrays,
    such that each has bins_per_array time bins. If the last chunk
    has fewer than bins_per_array bins, copy the last bins_per_array
    columns from the end of the original array.

    Returns:
        split_array : numpy.ndarray
            3D array with shape (num_samples, num_rows, bins_per_array).
    """
    # compute total number of 2D arrays to pre-allocate
    total_bins = array.shape[1]
    num_2d_arrays = int(np.ceil(total_bins / bin_shift))

    split_array = np.zeros((num_2d_arrays, array.shape[0], bins_per_array), dtype=array.dtype)

    for i in trange(len(split_array)):
        # get next set of start and end bins
        start_bin = i * bin_shift
        end_bin = start_bin + bins_per_array

        # extract array from original and fill new split array
        extracted_chunk = array[:, start_bin:end_bin]
        split_array[i, :, :extracted_chunk.shape[1]] = extracted_chunk

    if total_bins % bins_per_array != 0: # fix when unevenly split
        # last array currently only filled partially
        # set last array in split_array to larger chunk from the end
        split_array[-1] = array[:, -bins_per_array:]

    return split_array

@numba.njit(parallel=True)
def split_numba(array, bins_per_array, bin_shift):
    """
    Does the same thing as split() but uses Numba for
    increased speed performance.

    Splits long 2D array into 3D array of multiple 2D arrays,
    such that each has bins_per_array time bins. If the last chunk
    has fewer than bins_per_array bins, copy the last bins_per_array
    columns from the end of the original array.

    Returns:
        split_array : numpy.ndarray
            3D array with shape (num_samples, num_rows, bins_per_array).
    """
    # compute total number of 2D arrays to pre-allocate
    total_bins = array.shape[1]
    num_2d_arrays = int(np.ceil(total_bins / bin_shift))

    split_array = np.zeros((num_2d_arrays, array.shape[0], bins_per_array), dtype=array.dtype)

    for i in numba.prange(len(split_array)): # run in parallel for increased speed
        # get next set of start and end bins
        start_bin = i * bin_shift
        end_bin = start_bin + bins_per_array

        # extract array from original and fill new split array
        extracted_chunk = array[:, start_bin:end_bin]
        split_array[i, :, :extracted_chunk.shape[1]] = extracted_chunk

    if total_bins % bins_per_array != 0: # fix when unevenly split
        # last array currently only filled partially
        # set last array in split_array to larger chunk from the end
        split_array[-1] = array[:, -bins_per_array:]

    return split_array

def scale_data(ftdata):
    """Subtract each frequency channel in 3D array by its median and
    divide each array by its global standard deviation. Perform
    this standardization in chunks to avoid a memory overload."""

    for arr in tqdm(ftdata):
        stddev = np.std(arr)

        # subtract median from each row (spectrum) and divide every 2D array by its global stddev
        arr -= np.median(arr, axis=1, keepdims=True)
        arr /= stddev

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
        rescaled_chunk /= stddev # divide every 2D array by its stddev

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

def plot_confusion_matrix(val_ftdata, val_labels, pred_probs, confusion_matrix_name, enable_numba=True):
    pred_labels = np.round(pred_probs)
    # print out scores of various metrics
    accuracy, precision, recall, fscore, conf_mat = print_metric(val_labels, pred_labels)
    ftdata_shape = val_ftdata[0, :, :, 0].shape

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

def get_slope_from_driftRate(frame):
    """Convert drift rate from Hz/s to slope in pixel units.
    Assumes frame has metadata attribute with drift rate."""
    drift_rate = frame.metadata['drift_rate']
    slope_pixels = drift_rate / (frame.df/frame.dt)
    return slope_pixels

def get_driftRate_from_slope(slopes, df, dt):
    """Converts array of slopes in pixel units to drift rates (Hz/s).
    Assumes that slope is run/rise since drift rate is run/rise."""
    drift_rate = slopes * (df/dt)
    return drift_rate

def regression_slope(ftdata):
    """Detect line in image using triangle threshold and Hough transform.
    Convert angle to slope in pixel units, to be later converted to drift
    rate with knowledge of the sampling time/sampling frequency."""

    warnings.filterwarnings('ignore', message='Mean of empty slice.')
    warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')

    try:
        # use triangle thresholding to remove most noisy bits
        thresholded_data = ftdata >= filters.threshold_triangle(ftdata)

        # remove small objects (salt and pepper noise)
        small_objects_removed = morphology.remove_small_objects(thresholded_data, min_size=5)

        # segment image based on connected components, assuming longest component is desired signal
        # @source Vincent Agnus, https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
        # segmented_data = measure.label(thresholded_data, connectivity=2)

        # if segmented_data.max() == 0: # if no connected components, fall back to using thresholded image
        #     largestCC = thresholded_data
        # else:
        #     largestCC = segmented_data == np.argmax(np.bincount(segmented_data.flat)[1:]) + 1

        # find line and angle using Hough transform
        # tested_angles = np.linspace(-np.pi/2, np.pi/2, 360)
        # h, theta, d = transform.hough_line(largestCC, theta=tested_angles)

        # # pick best line using peak in Hough transform; might have no peaks and be set to None
        # angles = transform.hough_line_peaks(h, theta, d, num_peaks=1, min_distance=10)[1]

        # # convert from angle to slope (negative because drift rate is run/rise when time is y-axis)
        # slope_pixels = np.tan(-angles[0]) if angles else 0

        x, y = np.where(small_objects_removed)
        slope_pixels, intercept, low_slope, high_slope = theilslopes(y, x, alpha=0.99)
    except:
        slope_pixels = low_slope = high_slope = 0

    return np.array([slope_pixels, low_slope, high_slope])