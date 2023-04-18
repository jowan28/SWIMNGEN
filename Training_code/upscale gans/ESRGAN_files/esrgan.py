# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.nn import depth_to_space
from tensorflow.keras import Model
from tensorflow.keras import Input
import tensorflow as tf
from keras import backend
from tensorflow.keras.layers import Layer, InputSpec
class Attention(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,axes={3: input_shape[-1]})
        self.built = True


    def call(self, x):
        def hw_flatten(x):
            return backend.reshape(x, shape=[backend.shape(x)[0], backend.shape(x)[1]*backend.shape(x)[2], backend.shape(x)[-1]])
        
        f = backend.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = backend.bias_add(f, self.bias_f)
        g = backend.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = backend.bias_add(g, self.bias_g)
        h = backend.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = backend.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = backend.softmax(s, axis=-1)  # attention map

        o = backend.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = backend.reshape(o, shape=backend.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class ESRGAN(object):
    @staticmethod
    def generator(scalingFactor, featureMaps, residualBlocks,
            leakyAlpha, residualScalar):
        # initialize the input layer
        inputs = Input((None, None, 4))
        xIn = Rescaling(scale=1.0/255, offset=0.0)(inputs)
        # pass the input through CONV => LeakyReLU block
        xIn = Conv2D(featureMaps, 9, padding="same")(xIn)
        xIn = LeakyReLU(leakyAlpha)(xIn)
        
        # construct the residual in residual block
        x = Conv2D(featureMaps, 3, padding="same")(xIn)
        x1 = LeakyReLU(leakyAlpha)(x)
        x1 = Add()([xIn, x1])
        x = Conv2D(featureMaps, 3, padding="same")(x1)
        x2 = LeakyReLU(leakyAlpha)(x)
        x2 = Add()([x1, x2])
        x = Conv2D(featureMaps, 3, padding="same")(x2)
        x3 = LeakyReLU(leakyAlpha)(x)
        x3 = Add()([x2, x3])
        x = Conv2D(featureMaps, 3, padding="same")(x3)
        x4 = LeakyReLU(leakyAlpha)(x)
        x4 = Add()([x3, x4])
        x4 = Conv2D(featureMaps, 3, padding="same")(x4)
        xSkip = Add()([xIn, x4])
        # scale the residual outputs with a scalar between [0,1]
        xSkip = Lambda(lambda x: x * residualScalar)(xSkip)
        
        # create a number of residual in residual blocks
        for blockId in range(residualBlocks-1):
            x = Conv2D(featureMaps, 3, padding="same")(xSkip)
            x1 = LeakyReLU(leakyAlpha)(x)
            x1 = Add()([xSkip, x1])
            x = Conv2D(featureMaps, 3, padding="same")(x1)
            x2 = LeakyReLU(leakyAlpha)(x)
            x2 = Add()([x1, x2])
            x = Conv2D(featureMaps, 3, padding="same")(x2)
            x3 = LeakyReLU(leakyAlpha)(x)
            x3 = Add()([x2, x3])
            x = Conv2D(featureMaps, 3, padding="same")(x3)
            x4 = LeakyReLU(leakyAlpha)(x)
            x4 = Add()([x3, x4])
            x4 = Conv2D(featureMaps, 3, padding="same")(x4)
            xSkip = Add()([xSkip, x4])
            xSkip = Lambda(lambda x: x * residualScalar)(xSkip)
          
        # process the residual output with a conv kernel
        x = Conv2D(featureMaps, 3, padding="same")(xSkip)
        x = Add()([xIn, x])
        # upscale the image with pixel shuffle
        x = Conv2D(featureMaps * (scalingFactor // 2), 3,
            padding="same")(x)
        x = tf.nn.depth_to_space(x, 2)
        x = LeakyReLU(leakyAlpha)(x)
        # upscale the image with pixel shuffle
        x = Conv2D(featureMaps, 3, padding="same")(x)
        #commented out for a x2 upscaling
        #x = tf.nn.depth_to_space(x, 2)
        x = LeakyReLU(leakyAlpha)(x)
        # get the output layer
        x = Conv2D(4, 9, padding="same", activation="tanh")(x)
        output = Rescaling(scale=127.5, offset=127.5)(x)
        # create the generator model
        generator = Model(inputs, output)
        # return the generator model
        return generator
    
    @staticmethod
    def discriminator(featureMaps, leakyAlpha, discBlocks):
        # initialize the input layer and process it with conv kernel
        inputs = Input((None, None, 4))
        x = Rescaling(scale=1.0/127.5, offset=-1)(inputs)
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = LeakyReLU(leakyAlpha)(x)

        # pass the output from previous layer through a CONV => BN =>
        # LeakyReLU block
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leakyAlpha)(x)
        # create a downsample conv kernel config
        downConvConf = {
            "strides": 2,
            "padding": "same",
        }
        # create a number of discriminator blocks
        for i in range(1, discBlocks):
            # first CONV => BN => LeakyReLU block
            x = Conv2D(featureMaps * (2 ** i), 3, **downConvConf)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)
            # second CONV => BN => LeakyReLU block
            x = Conv2D(featureMaps * (2 ** i), 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)
            
        #apply self attention
        attention_layer = Attention(x.shape[-1])
        attention_layer.build(x.shape)
        x = attention_layer(x)
        # process the feature maps with global average pooling
        x = GlobalAvgPool2D()(x)
        x = LeakyReLU(leakyAlpha)(x)
        # final FC layer with sigmoid activation function
        x = Dense(1, activation="sigmoid")(x)
        # create the discriminator model
        discriminator = Model(inputs, x)
        # return the discriminator model
        return discriminator
    