import tensorflow as tf


# Callback Function which stops training when accuracy reaches 98%.
class LossCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs={}):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_epoch_end(self, epoch, logs={}):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and accuracy is {:7.2f}.".format(
                epoch, logs["loss"], logs["accuracy"]
        ))


def save_modelckpt():
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="mymodel_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    )
    return callback_checkpoint
