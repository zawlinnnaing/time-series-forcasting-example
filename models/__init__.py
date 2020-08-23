import config as cfg
import tensorflow as tf
from tools.window_generator import WindowGenerator
from tools.file_utils import make_dir, is_dir_empty
import os


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

    checkpoint_dir = os.path.join(cfg.CHECKPOINT_PATH, model_name)
    checkpoint_path = os.path.join(checkpoint_dir, '{epoch:04d}.ckpt')

    make_dir(checkpoint_dir)

    callbacks = []

    if not is_dir_empty(checkpoint_dir):
        load_weight(model, checkpoint_dir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        verbose=1,
    )
    callbacks.append(cp_callback)

    if cfg.EARLY_STOPPING['enabled'] is True:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min')
        callbacks.append(early_stopping)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )

    history = model.fit(
        window.train,
        epochs=cfg.MAX_EPOCH,
        validation_data=window.val,
        callbacks=callbacks,
        verbose=2,
    )

    return history


def load_weight(model, checkpoint_dir):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest_checkpoint)
    print('Loaded from checkpoint ==> {}'.format(latest_checkpoint))
    return model


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
