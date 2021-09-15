from functools import partial
import numpy as np

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D, Conv3D
from keras.layers.merge import concatenate
from keras.engine import Model
from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format("channels_first")
from metrics import weighted_dice_coefficient_loss, adaptive_dice_coefficient_loss



def discriminator(input_shape1, input_shape2 , n_filters = 16, kernel = (3, 3, 3)):        

        combined_imgs = concatenate([input_shape1, input_shape2] , axis=1)
        d1 = create_convolution_block (combined_imgs, n_filters = n_filters, instance_normalization=False )
        d2 = create_convolution_block(d1, n_filters = n_filters*2)
        d3 = create_convolution_block(d2, n_filters = n_filters*4)
        d4 = create_convolution_block(d3, n_filters = n_filters*8)
        validity = Conv3D(1, kernel, padding='same', strides=(1, 1, 1))(d4)
        return Model([input_shape1, input_shape2], validity)


def create_convolution_block(input_layer, n_filters , kernel=(3, 3, 3), activation=LeakyReLU,
                                    padding='same', strides=(2, 2, 2), instance_normalization=True): 
            layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
            if instance_normalization:
                from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization        
                layer = InstanceNormalization(axis=1)(layer)        
            return activation()(layer)

if __name__ == "__main__":
    img_A = Input((1, 128, 128, 128))
    img_B = Input((1, 128, 128, 128))
    model = discriminator(img_A, img_B, 32)    
    model.summary()