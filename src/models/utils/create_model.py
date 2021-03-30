import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.utils.building_blocks import *


def custom_CNN_model(input_shape, num_classes):

    '''
    Build custom CNN tensorflow model.

    Parameters:
        input_shape (tuple of ints): shape of the input images.
        num_classes (int): number of classes in which to classify images.

    Returns:
        model (tensorflow model): tensorflow model ready to compile and use.
    '''

    inputs = keras.Input(shape=input_shape)

    # Data augmentation block
    data_augmentation = keras.Sequential(
        [   
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    
    x = data_augmentation(inputs)

    # Data preprocessing block
    x = preprocessing_block(x,centercrop=True,rescaling=True)

    # Convolutional blocks

    x = convolutional_block(x,filters=256,name='conv1')
    x = convolutional_block(x,filters=128,name='conv2')
    x = convolutional_block(x,filters=64,name='conv3')

    # Dense layers block

    x = layers.Flatten(name="FlattenLayer")(x)

    x = layers.Dense(128,activation='relu',name='dense1')(x)
    x = layers.Dense(64,activation='relu',name='dense2')(x)

    # Output block

    if num_classes == 2:
        final_activation = "sigmoid"
        final_units = 1
    else:
        final_activation = "softmax"
        final_units = num_classes


    dropout = 0.3

    if dropout>0:
        x = layers.Dropout(dropout, name="DropoutLayer")(x)

    outputs = layers.Dense(final_units, activation=final_activation, name="OutputLayer")(x)

    model = keras.Model(inputs, outputs)

    return model
