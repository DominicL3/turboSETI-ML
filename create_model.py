import numpy as np
import matplotlib.pyplot as plt

import setigen as stg
from blimpy import Waterfall
import utils

from tqdm import tqdm
from time import time
import os, sys
import copy
import numba # speed up NumPy

# neural net imports
from tensorflow.keras.models import load_model
from model import construct_conv2d

"""
Train a Keras model to do binary classification on simulated pulses
vs. background RFI and save the best model from training. Exit automatically
if validation loss doesn't improve after a certain number of epochs.

Takes in a .fil file to use as background RFI and uses setigen to simulate
narrowband signals with randomly generated signal properties.

@source Liam Connor (https://github.com/liamconnor/single_pulse_ml)
@source Bryan Brzycki (https://github.com/bbrzycki/setigen)
"""

print(f'Array shape after splitting: {split_array.shape}')

start_time = time() # training time
model.fit(x=train_ftdata, y=train_labels,
          validation_data=(val_ftdata, val_labels),
          class_weight={0: 1, 1: weight_signal}, batch_size=32, epochs=32)

print(f"Training on {len(train_labels)} samples took took me {np.round((time() - start_time) / 60, 2)} minutes")

# print out scores of various metrics
accuracy, precision, recall, fscore, conf_mat = utils.print_metric(eval_labels, y_pred)

TP, FP, TN, FN = utils.get_classification_results(eval_labels, y_pred)

# get lowest confidence selection for each category
if TP.size:
    TPind = TP[np.argmin(y_pred_prob[TP])]  # Min probability True positive candidate
    TPdata = eval_ftdata[..., 0][TPind]
else:
    TPdata = np.zeros((NFREQ, NTIME))

if FP.size:
    FPind = FP[np.argmax(y_pred_prob[FP])]  # Max probability False positive candidate
    FPdata = eval_ftdata[..., 0][FPind]
else:
    FPdata = np.zeros((NFREQ, NTIME))

if FN.size:
    FNind = FN[np.argmax(y_pred_prob[FN])]  # Max probability False negative candidate
    FNdata = eval_ftdata[..., 0][FNind]
else:
    FNdata = np.zeros((NFREQ, NTIME))

if TN.size:
    TNind = TN[np.argmin(y_pred_prob[TN])]  # Min probability True negative candidate
    TNdata = eval_ftdata[..., 0][TNind]
else:
    TNdata = np.zeros((NFREQ, NTIME))

# plot the confusion matrix and display
plt.ioff()
plt.subplot(221)
plt.gca().set_title('TP: {}'.format(conf_mat[0][0]))
plt.imshow(TPdata, aspect='auto', interpolation='none')
plt.subplot(222)
plt.gca().set_title('FP: {}'.format(conf_mat[0][1]))
plt.imshow(FPdata, aspect='auto', interpolation='none')
plt.subplot(223)
plt.gca().set_title('FN: {}'.format(conf_mat[1][0]))
plt.imshow(FNdata, aspect='auto', interpolation='none')
plt.subplot(224)
plt.gca().set_title('TN: {}'.format(conf_mat[1][1]))
plt.imshow(TNdata, aspect='auto', interpolation='none')
plt.tight_layout()

# save data, show plot
print("Saving confusion matrix to {}".format(confusion_matrix_name))
plt.savefig(confusion_matrix_name, dpi=100)
plt.show()