import tensorflow as tf
from v1 import Yolov1


def test(split_size=7, num_boxes=2, num_classes=20):
    model = Yolov1(split_size=split_size, num_boxes=num_boxes,
                   num_classes=num_classes)
    x = tf.random.normal([2, 448, 448, 3])
    print(f"SHAPE = {model(x).shape}")


if __name__ == "__main__":
    test()
