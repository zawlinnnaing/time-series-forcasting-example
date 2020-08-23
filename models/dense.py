import tensorflow as tf

keras = tf.keras

dense = keras.Sequential([
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1)
])

multi_step_dense = keras.Sequential([
    # Shape: (time,features) => (time * features)
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1),  # shape: (output)
    # Re-add time step to output
    keras.layers.Reshape([1, -1])  # shape: (1, output)
])


def multi_output_dense(output_steps, num_features):
    return tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, :-1, :]),
        tf.keras.layers.Dense(units=1024, activation='relu'),
        # Shape [batch, 1, output_steps * num_features]
        tf.keras.layers.Dense(units=(output_steps * num_features), kernel_initializer=tf.initializers.zeros),
        # Shape: [batch, output_steps, num_features]
        tf.keras.layers.Reshape([output_steps, num_features])
    ])
