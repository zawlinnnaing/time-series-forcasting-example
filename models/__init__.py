import config as cfg
import tensorflow as tf
from tools.window_generator import WindowGenerator


def compile_and_fit(model: tf.keras.Model, window: WindowGenerator,
                    model_name: str,
                    patience=cfg.EARLY_STOPPING['patience'],
                    ):
    """
    Train model
    @param model_name:
    @param model:
    @param window:
    @param patience:
    @return:
    """
    early_stopping = None
    if cfg.EARLY_STOPPING['enabled'] is True:
        early_stopping = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min')]

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    history = model.fit(
        window.train,
        epochs=cfg.MAX_EPOCH,
        validation_data=window.val,
        callbacks=early_stopping,
        verbose=2,
    )

    return history


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        delta = self.model(inputs, training, mask)
        # The prediction for each time-step is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta
