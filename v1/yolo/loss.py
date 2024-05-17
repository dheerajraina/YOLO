import tensorflow as tf
import keras
import numpy as np
from v1 import intersection_over_union


class YoloLoss(tf.Module):
    def __init__(self, split_size, num_boxes, num_classes):
        super(YoloLoss, self).__init__()
        self.mse = keras.losses.MeanSquaredError(reduction="sum")
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def __call__(self, predictions, target):
        predictions = predictions.reshape(-1, self.split_size,
                                          self.split_size, self.num_classes+self.num_boxes*5)
        iou_b1 = intersection_over_union(
            predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(
            predictions[..., 26:30], target[..., 21:25])

        ious = tf.concat([tf.expand_dims(iou_b1, axis=0),
                         tf.expand_dims(iou_b2, axis=0)], axis=0)
        iou_maxes = tf.math.reduce_max(ious, axis=0)
        best_box = tf.argmax(ious, axis=0)
        exists_box = tf.expand_dims(target[..., 20], axis=3)  # Iobj_i

        ''' ----------------------For Box Coordinates ----------------------'''
        box_predictions = exists_box*(
            best_box*predictions[..., 26:30]
            + (1-best_box)*predictions[..., 21:25]
        )
        box_targets = exists_box*target[..., 21:25]

        box_predictions[..., 2:4] = tf.sign(box_predictions[..., 2:4])*tf.sqrt(
            tf.abs(box_predictions[..., 2:4]+1e-6)
        )

        box_targets[..., 2:4] = tf.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            np.reshape(box_predictions, newshape=(
                box_predictions.shape[0],)+(-1,)),
            np.reshape(box_targets, newshape=(
                box_targets.shape[0],)+(-1,))
        )

        ''' ----------------------For Object Loss ----------------------'''
        pred_box = (
            best_box*predictions[...:25, 26] +
            (1-best_box)*predictions[..., 20:21]
        )

        object_loss = self.mse(
            keras.layers.Flatten()(exists_box*pred_box),
            keras.layers.Flatten()(exists_box*target[..., 20:21])
        )

        ''' ----------------------For No Object Loss ----------------------'''
        input1 = (1-exists_box)*predictions[..., 20:21]
        input2 = (1-exists_box)*target[..., 20:21]
        no_object_loss = self.mse(
            np.reshape(input1, newshape=(input1.shape[0], -1)),
            np.reshape(input2, newshape=(input2.shape[0], -1)),
        )
        input1 = (1-exists_box)*predictions[..., 25:26]

        no_object_loss += self.mse(
            np.reshape(input1, newshape=(input1.shape[0], -1)),
            np.reshape(input2, newshape=(input2.shape[0], -1)),
        )

        ''' ----------------------For Class Loss ----------------------'''

        input1 = (exists_box)*predictions[..., :20]
        input2 = (exists_box)*target[..., :20]
        class_loss = self.mse(
            np.reshape(input1, newshape=(
                input1.shape[0],)+(-1,)),
            np.reshape(input2, newshape=(
                input2.shape[0],)+(-1,))
        )

        loss = (
            # represents first two rows of the loss function as present in the paper
            self.lambda_coord*box_loss
            + object_loss
            + self.lambda_noobj
            + class_loss
        )
        return loss
