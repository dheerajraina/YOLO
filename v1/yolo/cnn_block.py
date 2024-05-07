import tensorflow as tf
import keras


class CNNBlock(tf.Module):
    def __init__(self,  out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = keras.layers.Conv2D(
            filters=out_channels, use_bias=False, padding='same', **kwargs)
        self.batchnorm = keras.layers.BatchNormalization()
        self.leakyrelu = keras.activations.leaky_relu(negative_slope=0.1)

    def __call__(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
