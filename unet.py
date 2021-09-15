from keras.layers.merge import concatenate
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.optimizers import Adam
K.set_image_data_format("channels_first")


def unet_model_3d(input_shape=(1, 128, 128, 128), pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=5e-4, deconvolution=False,
                  depth=5, n_base_filters=16, loss_function='mae', activation_name="tanh"):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(
            input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth))
        layer2 = create_convolution_block(
            input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = UpSampling3D(size=pool_size)(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
#   act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=final_convolution)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function)
    return model


def create_convolution_block(input_layer, n_filters,  kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):

    layer = Conv3D(n_filters, kernel, padding=padding,
                   strides=strides)(input_layer)
    from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
    layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)
