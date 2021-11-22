"""
Author: Sam Gagnon-Hartman, modified by Jonathan Colaco Carr 
                                        (jonathan.colacocarr@mail.mcgill.ca)

Python script to build the U-Net architecture proposed in Isensee et al. 2017
"""

import logging
import typing
from typing import Optional, List, Any
import tensorflow as tf
import tensorflow.keras.backend as K
from functools import partial

from tensorflow import pad
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import (Layer, 
                                     LeakyReLU,
                                     Add,
                                     UpSampling3D,
                                     Activation,
                                     SpatialDropout3D,
                                     Conv3D,
                                     BatchNormalization,
                                     Concatenate)


from loss_functions import (dice_coefficient_loss, 
                            TverskyLoss,
                            FocalTverskyLoss,
                            tmp_weighted_crossentropy,
                            weighted_crossentropy)


def init_logger(f, name, level=logging.INFO):
    """Instantiates logger :name: and sets logfile to :f:"""
    logger = logging.getLogger(name)

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s: %(levelname).1s %(filename)s:%(lineno)d] %(message)s")
    file_handler = logging.FileHandler(f)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

LOGGER = init_logger("test.log", __name__)

KWARGS = {} # {"kernel_initializer": HeNormal()}

class ReflectionPadding3D(Layer):
    """
    3D Reflection Padding keras Layer. This class implements a tensor-in
    tensor-out computation in the call() function, and holds some state
    described by Tensorflow Variables.
    ----------
    Attributes
    :padding: tuple (padding_width, padding_height)
    :**kwargs: arguments for the keras Layer object
    """

    def __init__(self,
                 padding: tuple =(1, 1, 1),
                 **kwargs):

        self.padding = tuple(padding)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, 
                             input_shape: tuple) -> tuple:
        """Computes shape of output tensor after padding"""
        return (input_shape[0], \
                input_shape[1] + 2 * self.padding[0], \
                input_shape[2] + 2 * self.padding[1], \
                input_shape[3] + 2 * self.padding[2], input_shape[4])

    def call(self,
             input_tensor: tf.Tensor,
             mask=None) -> tf.Tensor:
        """Returns input tensor with extra padding"""
        padding_width, padding_height, padding_depth = self.padding

        return pad(input_tensor, 
                   [[0,0], 
                    [padding_height, padding_height], 
                    [padding_width, padding_width],
                    [padding_width, padding_width], 
                    [0,0]], 
                   'REFLECT')


def create_localization_module(input_layer: Layer, 
                               n_filters: int):

    """Localization module passes an upsampled tensor through a 3x3x3
    convolution. The second 1x1x1 convolution is to reduce the feature map for 
    memory efficiency."""
    layer1 = ReflectionPadding3D()(input_layer)
    convolution1 = create_convolution_block(layer1, n_filters)
    convolution2 = create_convolution_block(convolution1, 
                                            n_filters, 
                                            kernel=(1, 1, 1))

    return convolution2


def create_up_sampling_module(input_layer: Layer,
                              n_filters: int,
                              size: Optional[int] = (2, 2, 2)):

    up_sample = UpSampling3D(size=size)(input_layer)
    layer1 = ReflectionPadding3D()(up_sample)
    convolution = create_convolution_block(layer1, n_filters)

    return convolution


def create_context_module(input_layer: Layer,
                          n_level_filters: int,
                          dropout_rate: Optional[float] = 0.3,
                          data_format: Optional[str] = "channels_last"):

    layer1 = ReflectionPadding3D()(input_layer)
    convolution1 = create_convolution_block(input_layer=layer1, 
                                            n_filters=n_level_filters)

    dropout = SpatialDropout3D(rate=dropout_rate, 
                               data_format=data_format)(convolution1)

    layer2 = ReflectionPadding3D()(dropout)
    convolution2 = create_convolution_block(input_layer=layer2, 
                                            n_filters=n_level_filters)

    return convolution2


def create_convolution_block(input_layer: Layer,
                             n_filters: int,
                             batch_normalization: Optional[bool] = False,
                             kernel: Optional[tuple] = (3, 3, 3),
                             activation: Optional[Any] = None,
                             padding: Optional[str] = 'valid',
                             strides: Optional[tuple] = (1, 1, 1),
                             instance_normalization: Optional[bool] = False):
    """
    Creates a 3D convolutional Layer followed by BatchNormalization (if
    required), InstanceNormalization (if required). 
    -------
    Returns
    Non-linear activation of output layer.
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides,
                   **KWARGS)(input_layer)
    LOGGER.info(layer.shape)
    if batch_normalization:
        layer = BatchNormalization()(layer)
    elif instance_normalization:
        layer = InstanceNormalization()(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


create_convolution_block = partial(create_convolution_block, 
                                   activation=LeakyReLU, 
                                   instance_normalization=True)


def isensee2017_model(inputs, 
                      n_base_filters: Optional[int] = 16,
                      depth: Optional[int] = 5,
                      dropout_rate: Optional[float] = 0.3,
                      n_segmentation_levels: Optional[int] = 3,
                      n_labels: Optional[int] = 1,
                      optimizer: Optional[Any] = Adam,
                      initial_learning_rate: Optional[float] = 5e-4,
                      loss_function: Optional[Any] = dice_coefficient_loss,
                      activation_name: Optional[str] = "sigmoid"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    """

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    # Create downwards path
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            layer = ReflectionPadding3D()(current_layer)
            in_conv = create_convolution_block(layer, n_level_filters)
        else:
            layer = ReflectionPadding3D()(current_layer)
            in_conv = create_convolution_block(layer,
                                               n_level_filters,
                                               strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv,
                                                     n_level_filters,
                                                     dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    LOGGER.debug(f"level filters: {level_filters}")

    # Create upwards path
    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):

        up_sampling = create_up_sampling_module(current_layer,
                                                level_filters[level_number])

        concatenation_layer = Concatenate()([level_output_layers[level_number],
                                             up_sampling])

        localization_output = create_localization_module(concatenation_layer, 
                                                         level_filters[level_number])

        current_layer = localization_output

        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):

        segmentation_layer = segmentation_layers[level_number]

        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate),
                  loss=dice_coefficient_loss)

    return model
