import tensorflow as tf
import keras


class CNNBlock(keras.layers.Layer):
    def __init__(self,  out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = keras.layers.Conv2D(
            filters=out_channels, use_bias=False, **kwargs)
        self.batchnorm = keras.layers.BatchNormalization()
        self.leakyrelu = keras.layers.LeakyReLU(negative_slope=0.1)

    def call(self, x):
        x = self.conv(x)
        return self.leakyrelu(self.batchnorm(x))
