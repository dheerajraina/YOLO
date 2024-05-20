import keras
from v1 import testRunner, Yolov1, YoloLoss, train_fn, hyperparameters, miscellaneous

if __name__ == "__main__":
    testRunner()
    model = YoloLoss(split_size=7, num_boxes=2, num_classes=20)
    optimizer = keras.optimizers.Adam(
        learning_rate=hyperparameters["LEARNING_RATE"]
    )
    loss_fn = YoloLoss(split_size=7, num_boxes=2, num_classes=20)

    for epoch in range(hyperparameters["EPOCHS"]):
        print(f"EPOCH{epoch}")
