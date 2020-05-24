# neural net imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D

from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def construct_conv2d(num_conv_layers=2, num_filters=32, n_dense1=256, n_dense2=128,
                    saved_model_name='best_model.h5', previous_model_to_train=None):
    """
    Parameters:
    ----------
    num_conv_layers : int
        Number of convolutional layers to implement (MAX 4 due to pooling layers,
        otherwise Keras will throw an error)
    num_filters : int
        Number of filters in first convolutional layer, doubles after each convolutional block.

    Returns
    -------
    cnn_2d : Keras model
        Model to be used on frequency-time data
    """

    if previous_model_to_train is not None:
        print("Loading in previous model: " + previous_model_to_train)
        cnn_2d = load_model(previous_model_to_train, compile=False)
    else:
        cnn_2d = Sequential()

        # create num_filters convolution filters, each of size 3x3
        cnn_2d.add(Conv2D(num_filters, (3, 3), padding='same', input_shape=(None, None, 1), name='conv2d_1'))
        cnn_2d.add(BatchNormalization(name='batch_norm_1')) # standardize all inputs to activation function
        cnn_2d.add(Activation('relu', name='relu_1'))
        cnn_2d.add(MaxPooling2D(pool_size=(2, 2), name='max_pool_1')) # max pool to reduce the dimensionality

        # repeat and double the filter size for each convolutional block to make this DEEP
        for layer_number in np.arange(2, num_conv_layers + 1):
            num_filters *= 2
            cnn_2d.add(Conv2D(num_filters, (3, 3), padding='same', name=f'conv2d_{layer_number}'))
            cnn_2d.add(BatchNormalization(name=f'batch_norm_{layer_number}'))
            cnn_2d.add(Activation('relu', name=f'relu_{layer_number}'))
            cnn_2d.add(MaxPooling2D(pool_size=(2, 2), name=f'max_pool_{layer_number}'))

        # max pool all feature maps
        cnn_2d.add(GlobalMaxPooling2D(name='global_max_pool'))

        # run through two fully connected layers
        fc_1 = cnn_2d.add(Dense(n_dense1, activation='relu', name='fc_1'))
        dropout_1 = cnn_2d.add(Dropout(0.4, name='dropout_1'))

        fc_2 = cnn_2d.add(Dense(n_dense2, activation='relu', name='fc_2'))
        dropout_2 = cnn_2d.add(Dropout(0.3, name='dropout_2'))

        # predict what the label should be
        pred_layer = cnn_2d.add(Dense(1, activation='sigmoid', name='sigmoid_output'))

    # optimize using Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # save model with lowest validation loss
    loss_callback = ModelCheckpoint(saved_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    # cut learning rate in half if validation loss doesn't improve in 5 epochs
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # stop training if validation loss doesn't improve after 15 epochs
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    return cnn_2d