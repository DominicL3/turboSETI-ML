# neural net imports
import tensorflow
tensorflow.compat.v1.disable_eager_execution()

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D

from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def build_branch(input_layer, num_conv_layers=2, num_filters=32, n_dense1=256, n_dense2=128):
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
    model_branch : Keras model
        Model to be used on frequency-time data
    """
    # create num_filters convolution filters, each of size 3x3
    model_branch = Conv2D(num_filters, (3, 3), padding='same', input_shape=(None, None, 1))(input_layer)
    model_branch = BatchNormalization()(model_branch) # standardize all inputs to activation function
    model_branch = Activation('relu')(model_branch)
    model_branch = MaxPooling2D(pool_size=(2, 2))(model_branch) # max pool to reduce the dimensionality

    # repeat and double the filter size for each convolutional block to make this DEEP
    for layer_number in range(2, num_conv_layers + 1):
        num_filters *= 2
        model_branch = Conv2D(num_filters, (3, 3), padding='same')(model_branch)
        model_branch = BatchNormalization()(model_branch)
        model_branch = Activation('relu')(model_branch)
        model_branch = MaxPooling2D(pool_size=(2, 2))(model_branch)

    # max pool all feature maps
    model_branch = GlobalMaxPooling2D()(model_branch)

    # run through two fully connected layers
    model_branch = Dense(n_dense1, activation='relu')(model_branch)
    model_branch = Dropout(0.4)(model_branch)

    model_branch = Dense(n_dense2, activation='relu')(model_branch)
    model_branch = Dropout(0.3)(model_branch)

    return model_branch

def construct_conv2d(num_conv_layers=2, num_filters=32, n_dense1=256, n_dense2=128,
                    saved_model_name='best_model.h5', previous_model=None):

    if previous_model is not None:
        print("Loading in previous model: " + previous_model)
        model = load_model(previous_model, compile=False)
    else:
        input_layer = Input(shape=(None, None, 1))

        # predict what the label should be
        class_branch = build_branch(input_layer, num_conv_layers, num_filters, n_dense1, n_dense2)
        class_branch = Dense(1, activation='sigmoid', name='class_output')(class_branch)

        slope_branch = build_branch(input_layer, num_conv_layers, num_filters, n_dense1, n_dense2)
        slope_branch = Dense(1, activation='linear', name='slope_output')(slope_branch)

        model = Model(inputs=input_layer, outputs=[class_branch, slope_branch], name=saved_model_name)

    return model

def fit_model(model, train_ftdata, train_labels, val_ftdata, val_labels,
                train_slopes, val_slopes, saved_model_name='best_model.h5',
                weight_signal=1.0, batch_size=32, epochs=32, classification_loss_weight=10):
    """Fit a model using the given training data and labels while
    validating each epoch. Continually save the model that improves
    on the best val_loss."""

    # define loss for classification and regression and weight each loss
    # classification is more important, so its weight should be > 1
    loss_dict = {'class_output': 'binary_crossentropy', 'slope_output': 'mean_squared_error'}
    loss_weights_dict = {'class_output': classification_loss_weight, 'slope_output': 1}

    # compile model and optimize using Adam
    model.compile(loss=loss_dict, loss_weights=loss_weights_dict,
                    optimizer='adam', metrics=['accuracy'])

    # save model with lowest validation loss
    loss_callback = ModelCheckpoint(saved_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    # cut learning rate in half if validation loss doesn't improve in 5 epochs
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # stop training if validation loss doesn't improve after 15 epochs
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

    model.fit(x=train_ftdata, y={'class_output': train_labels, 'slope_output': train_slopes},
            validation_data=(val_ftdata, {'class_output': val_labels, 'slope_output': val_slopes}),
            class_weight={0: 1, 1: weight_signal}, batch_size=batch_size, epochs=epochs,
            callbacks=[loss_callback, reduce_lr_callback, early_stop_callback])