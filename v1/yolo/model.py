import tensorflow as tf
import keras
from v1.utils import architecture
from v1.yolo.cnn_block import CNNBlock
import numpy as np


class Yolov1(tf.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcl = self.create_fc_layers(**kwargs)

    def __call__(self, x):
        x = self.darknet(x)
        x = keras.layers.Flatten()(x)
        return self.fcl(x)

    def create_conv_layers(self, architecture):
        layers = keras.models.Sequential()
        for x in architecture:
            if type(x) == tuple:
                layers.add(keras.layers.ZeroPadding2D(x[3]))
                layers.add(
                    CNNBlock(
                        out_channels=x[1],
                        kernel_size=x[0],
                        strides=x[2]
                    )
                )
            elif type(x) == str:
                layers.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers.add(keras.layers.ZeroPadding2D(x[3]))
                    layers.add(
                        CNNBlock(
                            conv1[1],
                            kernel_size=conv1[0],
                            strides=conv1[2]
                        )
                    )
                    layers.add(keras.layers.ZeroPadding2D(x[3]))
                    layers.add(
                        CNNBlock(
                            conv2[1],
                            kernel_size=conv2[0],
                            strides=conv2[2]
                        )
                    )

        return layers

    def create_fc_layers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(496),  # original paper usee 4960 units
            keras.layers.Dropout(0),
            keras.layers.LeakyReLU(negative_slope=0.1),
            keras.layers.Dense(S*S*(C+B*5))
        ])
