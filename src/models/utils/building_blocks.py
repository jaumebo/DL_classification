import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def convolutional_block(x,
                        filters,
                        name,
                        kernel_size=3,
                        padding="same",
                        pooling="max",
                        pool_size=(2, 2),
                        strides=None,
                        activation="relu",
                        batch_norm = False
                        ):

    '''
    Returns the output tensor of a CNN block given an input tensor x.

    Parameters:
        x (tensor): the entry tensor.
        filters (int): number of convolutional filters.
        name (str): name of the CNN block.
        kernel_size (int): size of the squared kernel of the convolutional filter (kernel_size,kernel_size).
        padding ("valid" or "same"): "valid" -> no padding, "same" -> padding such that output has the same dimensions as the input.
        pooling ("max","average" or None): "max" -> Apply MaxPooling2D, "average" -> Apply AveragePooling2D, None -> apply no pooling.
        pool_size (int or tuple of 2 ints): factors by which to downscale (vertical, horizontal). If only one integer is specified, the same window length will be used for both dimensions.
        strides (int or tuple of 2 ints or None): Strides values. If None, it will default to pool_size.
        activation (str): activation function. See available functions at: https://keras.io/api/layers/activations/#available-activations
        batch_norm (bool): whether to apply batch normalization or not.

    Returns:
        x (tensor): Tensor outputs of original tensor x passing through the defined CNN block.
    '''
    
    x = layers.Conv2D(filters,kernel_size,padding=padding,name="Conv2D_"+name)(x)
    
    if activation=='relu':
        x = layers.Activation('relu',name="ReLU_"+name)(x)

    if pooling=='max':
        x = layers.MaxPooling2D(pool_size=pool_size,strides=strides,name="MaxPooling2D_"+name)(x)
    elif pooling=='average':
        x = layers.AveragePooling2D(pool_size=pool_size,strides=strides,name="MaxPooling2D_"+name)(x)
    
    if batch_norm=='true':
        x = layers.BatchNormalization(name="BatchNorm_"+name)(x)

    return x

def preprocessing_block(x,
                        centercrop=False,
                        crop_x=256,
                        crop_y=256,
                        rescaling=True,
                        rescale_value=(1.0/255)
                        ):

    '''
    Returns the output tensor of a data preprocessing block given an input tensor x. The preprocessing is done inside the model.

    Parameters:
        x (tensor): the entry tensor.
        centercrop (bool): whether to apply CenterCrop or not.
        crop_x (int): center crop x size.
        crop_y (int): center crop y size.
        rescaling (bool): whether to apply rescaling or not.
        rescale_value (float): value to which rescale pixel values -> pixel_values * rescale_value

    Returns:
        x (tensor): Tensor outputs of original tensor x passing through the defined preprocessing block.
    '''

    if centercrop:
        x = layers.experimental.preprocessing.CenterCrop(crop_x,crop_y)(x)

    if rescaling:
        x = layers.experimental.preprocessing.Rescaling(rescale_value)(x)
    
    return x