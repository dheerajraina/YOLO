import tensorflow as tf
from tqdm import tqdm


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch, (x, y) in enumerate(loop):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_fn(predictions, y)
            mean_loss.append(loss.numpy())
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # update progress bar
        tqdm.write(f'Batch {batch}, Loss: {loss.numpy()}')

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
