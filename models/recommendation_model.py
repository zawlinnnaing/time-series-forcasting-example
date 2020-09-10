import tensorflow as tf

embedding_dim = 50


def matrix_factorization_model(n_products, n_customers):
    product_input = tf.keras.layers.Input([1], name='product_input')  # shape(batch, 1)
    product_embedding = tf.keras.layers.Embedding(
        n_products + 1,
        embedding_dim,
        name='product_embeddings'
    )(product_input)  # shape (batch, 1, n_embedding_dim)
    product_vector = tf.keras.layers.Flatten(name='product_vector')(product_embedding)  # shape(batch, n_embedding_dim)

    customer_input = tf.keras.layers.Input([1], name='customer_input')
    customer_embedding = tf.keras.layers.Embedding(
        n_customers + 1,
        embedding_dim,
        name='customer_embeddings'
    )(customer_input)  # shape: (batch, 1, n_embedding)
    customer_vector = tf.keras.layers.Flatten(name='customer_vector')(
        customer_embedding)  # shape: (batch, n_embedding_dim)

    dot_product = tf.keras.layers.Dot(axes=1, name='vector_dot_prod')([product_vector, customer_vector])
    model = tf.keras.Model([product_input, customer_input], dot_product, name='matrix_factorization')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])
    return model


def deep_nn_model(product_feature_dim, customer_feature_dim):
    # Note: batch_size will be omitted for following shapes. Assume that batch_size is first dimension of shape.
    # i.e (batch_size, shape_1, shape_2)

    # Shape: (,total_feature)
    input_layer = tf.keras.layers.Input((product_feature_dim + customer_feature_dim, 1), name="model_input")
    # input_layer = tf.keras.layers.Permute((2, 1),
    #                                       input_shape=(product_feature_dim + customer_feature_dim, 1),
    #                                       name="input_permute_layer")
    print('input_layer', input_layer.shape)
    # Shape: (,product_feature_dim)
    product_input = tf.keras.layers.Lambda(lambda x: x[:, :product_feature_dim, :], name="product_input")(input_layer)
    # Shape: (,product_feature_dim, embedding_dim)
    product_embedding = tf.keras.layers.Dense(
        embedding_dim,
        name="product_embeddings"
    )(product_input)
    # Shape: (embedding_dim, 64)
    product_dense_1 = tf.keras.layers.Dense(64, activation='relu')(product_embedding)
    print("product shapes: ", product_input.shape, product_embedding.shape, product_dense_1.shape)

    # Shape: (customer_feature_dim)
    customer_input = tf.keras.layers.Lambda(lambda x: x[:, -customer_feature_dim:, :], name="customer_input")(
        input_layer)

    # Shape: (customer_feature_dim, embedding_dim)
    customer_embedding = tf.keras.layers.Dense(
        embedding_dim,
        name="customer_embeddings"
    )(customer_input)
    # Shape: (
    customer_dense_1 = tf.keras.layers.Dense(64, activation='relu')(customer_embedding)
    print("customer shapes: ", customer_input.shape, customer_embedding.shape, customer_dense_1.shape)
    concatted_layer = tf.keras.layers.Concatenate(axis=1, name="concatted_layer")([product_dense_1, customer_dense_1])
    output = tf.keras.layers.Dense(1, activation="relu")(concatted_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output, name="DNNRecommendationModel")
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mse', 'mae'])
    return model
