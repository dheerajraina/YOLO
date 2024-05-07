import tensorflow as tf
import keras
from v1.utils import constants
from cnn_block import CNNBlock



class Yolov1(tf.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = constants.architecture
        self.in_channels = in_channels,
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcl = self.create_fc_layers(**kwargs)

    def __call__(self, x):
        x = self.darknet(x)
        return self.fcl(keras.layers.Flatten(x))

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        out_channels=x[1],
                        kernel_size=x[0],
                        strides=x[2]
                    )
                ]
            elif type(x) == str:
                layers += [keras.layers.MaxPool2D(pool_size=2, strides=2)]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            conv1[1],
                            kernel_size=conv1[0],
                            strides=conv1[2]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv2[1],
                            kernel_size=conv2[0],
                            strides=conv2[2]
                        )
                    ]

        return keras.models.Sequential(*layers)

    def create_fc_layers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(496),  # original paper usee 4960 units
            keras.layers.Dropout(0),
            keras.activations.leaky_relu(negative_slope=0.1),
            keras.layers.Dense(S*S*(C+B*S))
        ])
