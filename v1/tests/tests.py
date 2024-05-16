
import tensorflow as tf
from v1 import CNNBlock, Yolov1, custom_assert
import numpy as np


def testRunner():
    test_cnn_block()
    test_yolo_model()


def test_cnn_block():
    try:
        cnn_block = CNNBlock(10, kernel_size=7,
                             strides=2)

        cnn_block(np.random.rand(1, 10, 20, 3))
    except Exception as e:
        raise (e)
    else:
        print("Test Case 1 passed")


def test_yolo_model(split_size=7, num_boxes=2, num_classes=20):
    try:
        model = Yolov1(split_size=split_size, num_boxes=num_boxes,
                       num_classes=num_classes)
        x = tf.random.normal([2, 448, 448, 3])
        custom_assert(model(x).shape == (2, 1470),
                      "Model doesn't return desired shape")
    except Exception as e:
        raise (e)
    else:
        print("Test Case 2 passed")
