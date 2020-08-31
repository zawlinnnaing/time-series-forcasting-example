import tensorflow as tf

embedding_dim = 50


def make_model(n_products, n_customers):
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
    return tf.keras.Model([product_input, customer_input], dot_product, name='matrix_factorization')
