import tensorflow as tf


class BaseLine(tf.keras.Model):
    """
    BaseLine model that predicts output as the same as input.
    """

    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def get_config(self):
        """
        Get config
        """
        pass

    def call(self, inputs, training=None, mask=None):
        """
        @param inputs:
        @param training:
        @param mask:

        """
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
