# neural net imports
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, add
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def build_CNN(input_layer, num_conv_layers=2, num_filters=32):
    """
    Build the convolutional layers, which will extract features from input spectrograms.

    Parameters:
    ----------
    num_conv_layers : int
        Number of convolutional layers to use in model.
    num_filters : int
        Number of filters in first convolutional layer. Doubles with each convolutional block.

    Returns
    -------
    cnn_2d : Keras tensor
        Output tensor of convolutional feature maps.
    """
    # create num_filters convolution filters, each of size 3x3
    cnn_2d = Conv2D(num_filters, (3, 3), padding='same', input_shape=(None, None, 1))(input_layer)
    cnn_2d = BatchNormalization()(cnn_2d) # standardize all inputs to activation function
    cnn_2d = Activation('relu')(cnn_2d)
    cnn_2d = MaxPooling2D(pool_size=(2, 2))(cnn_2d) # max pool to reduce dimensionality by half

    # repeat and double the filter size for each convolutional block to make this DEEP
    for layer_number in range(2, num_conv_layers + 1):
        num_filters *= 2
        cnn_2d = Conv2D(num_filters, (3, 3), padding='same')(cnn_2d)
        cnn_2d = BatchNormalization()(cnn_2d)
        cnn_2d = Activation('relu')(cnn_2d)
        # cnn_2d = MaxPooling2D(pool_size=(2, 2))(cnn_2d)

    # max pool all feature maps
    cnn_2d = GlobalMaxPooling2D()(cnn_2d)

    return cnn_2d

def build_FC(cnn_2d, inputs, n_dense1=256, n_dense2=128):
    """
    Build the fully-connected layers. One branch of FC layers uses the
    feature maps to predict the class of the input, the other branch of FC layers
    uses the same feature maps AND the outputted class prediction to regress the
    drift rate of the input (if class 0, predict 0 drift rate).

    Parameters:
    ----------
    cnn_2d : Keras tensor
        Input Keras CNN tensor (output of CNN layers). Feature maps are used
        as inputs to the fully-connected layers.
    inputs : list
        List of Keras inputs to feed to fully-connected layers.
    n_dense1 : int
        Number of neurons in first hidden layer.
    n_dense2 : int
        Number of neurons in second hidden layer.

    Returns
    -------
    fc_layers : Keras tensor
        Output tensor of two fully connected layers.
        Full model to predict both class and drift rate.
    """

    # run through two fully connected layers
    # add Dropout for regularization (mitigate overfitting)
    fc_layers = Dense(n_dense1, activation='relu')(cnn_2d)
    fc_layers = Dropout(0.2)(fc_layers)

    fc_layers = Dense(n_dense2, activation='relu')(fc_layers)
    fc_layers = Dropout(0.2)(fc_layers)

    return fc_layers

def construct_model(num_conv_layers=2, num_filters=32, n_dense1=256, n_dense2=128,
                    saved_model_name='best_model.h5', previous_model=None):
    """
    Construct the full model. Convolutional feature maps are passed to both
    fully-connected branches, where one predicts the class and the other
    predicts the drift rate while using the output of the class branch as
    a second input. Name the model its original save location.

    If previous_model is provided, load it in to continue training instead of
    building a new network from scratch.

    Parameters:
    ----------
    num_conv_layers : int
        Number of convolutional layers to use in model.
    num_filters : int
        Number of filters in first convolutional layer. Doubles with each convolutional block.
    n_dense1 : int
        Number of neurons in first hidden layer.
    n_dense2 : int
        Number of neurons in second hidden layer.
    saved_model_name : str, optional
        Path to save model, also name of the model.
    previous_model : str, optional
        Path to previously saved model.

    Returns
    -------
    model : Keras model
        Full model to predict both class and drift rate.
    """

    if previous_model is not None:
        print("Loading in previous model: " + previous_model)
        model = load_model(previous_model, compile=False)
    else:
        input_layer = Input(shape=(None, None, 1))

        # create convolutional feature extraction layers
        cnn_2d = build_CNN(input_layer, num_conv_layers, num_filters)

        # predict what the class label should be
        class_branch = build_FC(cnn_2d, input_layer, n_dense1, n_dense2)
        class_branch = Dense(1, activation='sigmoid', name='class')(class_branch)

        # predict SLOPE with input image AND predicted class (drift rate calculated later)
        # network doesn't have access to channel bandwidth and sampling time
        # double the number of hidden neurons since drift rate is harder to predict than class
        slope_branch = build_FC(cnn_2d, [input_layer, class_branch], n_dense1*2, n_dense2*2)
        slope_branch = Dense(1, activation='linear', name='slope')(slope_branch)

        model = Model(inputs=input_layer, outputs=[class_branch, slope_branch], name=saved_model_name)

    return model

def fit_model(model, train_ftdata, train_labels, val_ftdata, val_labels,
                train_slopes, val_slopes, saved_model_name='best_model.h5',
                weight_signal=1.0, classification_loss_weight=1e5, batch_size=32, epochs=32):
    """
    Fit a model using the given training data and labels while validating each epoch.
    Save the model only when it performs better than the current val_loss. Weights can
    be adjusted to penalize missing a true signal or for classification over regression
    (more import to classify correctly than to get the exact drift rate).

    Parameters:
    ----------
    model : Keras model
        Model to fit and save.
    train_ftdata, val_ftdata : numpy.ndarray
        Training and validation input arrays with shape (batch, height, width, channels=1)
    train_labels, val_labels : numpy.ndarray
        Classification labels corresponding to training and validation input data.
    train_slopes, val_slopes : numpy.ndarray
        Drift rates for training and validation input data. Arrays that have no signal
        in them should be labeled with drift rate of 0.
    saved_model_name : str
        Path to save model, also name of the model.
    weight_signal : float, optional
        Penalty for missing a true signal. Increases recall.
    classification_loss_weight : float, optional
        Penalty on classification over regression. Since slope regression uses MSE loss
        and classification uses binary_crossentropy, there is a disproportionate loss on
        regression. Increasing this parameter increases the weight on classification loss.
    batch_size : int, optional
        Batch size for training network.
    epochs : int, optional
        Number of maximum epochs to train for. Model will stop at an earlier epoch
        if val_loss does not improve after 20 epochs.
    """

    # define loss for classification and regression and weight each loss
    # classification is more important, so its weight should be > 1
    loss_dict = {'class': 'binary_crossentropy', 'slope': 'mean_squared_error'}
    loss_weights_dict = {'class': classification_loss_weight, 'slope': 1}

    # compile model and optimize using Adam
    model.compile(loss=loss_dict, loss_weights=loss_weights_dict,
                    optimizer='adam', metrics={'class':'accuracy', 'slope': 'mean_absolute_percentage_error'})

    # save model with lowest validation loss
    loss_callback = ModelCheckpoint(saved_model_name, monitor='val_loss', verbose=1, save_best_only=True)

    # cut learning rate in half if validation loss doesn't improve in 5 epochs
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # stop training if validation loss doesn't improve after 20 epochs
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    model.fit(x=train_ftdata, y={'class': train_labels, 'slope': train_slopes},
            validation_data=(val_ftdata, {'class': val_labels, 'slope': val_slopes}),
            class_weight={'class': {0: 1, 1: weight_signal}}, batch_size=batch_size, epochs=epochs,
            callbacks=[loss_callback, reduce_lr_callback, early_stop_callback])