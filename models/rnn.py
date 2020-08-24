import tensorflow as tf
from models import ResidualWrapper

layers = tf.keras.layers


def lstm_model(num_features):
    return tf.keras.Sequential([
        # Shape (batch, time, features) => (batch, time, lstm_units)
        layers.LSTM(units=32, return_sequences=True),
        # Shape (batch, time, features)
        layers.Dense(units=num_features)
    ])


def residual_lstm(num_features):
    return ResidualWrapper(lstm_model(num_features))


def single_shot_lstm(output_steps, num_features):
    return tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(128, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(output_steps * num_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([output_steps, num_features])
    ])


class FeedBack(tf.keras.Model):

    def __init__(self, units, output_steps, num_features):
        super().__init__()
        self.output_steps = output_steps
        self.num_features = num_features
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units=units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(units=num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        for i in range(1, self.output_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)

            predictions.append(prediction)
        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
