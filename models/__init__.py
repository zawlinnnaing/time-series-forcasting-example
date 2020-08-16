import config as cfg
import tensorflow as tf


def compile_and_fit(model: tf.keras.Model, window, patience=cfg.EARLY_STOPPING['patience']):
    """
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
        callbacks=early_stopping
    )

    return history
