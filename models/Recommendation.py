import pandas as pd
from models import load_weight
from tools.file_utils import check_is_csv, make_dir, is_dir_empty
from sklearn.model_selection import train_test_split
from models.matrix_factorization import make_model
import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector


class RecommendationModel:
    def __init__(self,
                 rating_path,
                 product_path,
                 customer_path,
                 checkpoint_dir='checkpoints/recommendation',
                 checkpoint_name='recommendation_candidate_model.ckpt',
                 load_checkpoint=True):
        if (not check_is_csv(rating_path)) or (not check_is_csv(product_path) or (not check_is_csv(customer_path))):
            raise Exception('Input path {}, {}, {} is not csv'.format(rating_path, product_path, customer_path))
        self.rating_df = pd.read_csv(rating_path)
        self.product_series = pd.read_csv(product_path)
        self.customer_series = pd.read_csv(customer_path)

        # Create checkpoint dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        self.load_checkpoint = load_checkpoint

        make_dir(checkpoint_dir)
        self.n_of_products, self.n_of_customers = self.product_series.size, self.customer_series.size
        self.train_ds, self.test_ds = train_test_split(self.rating_df)
        self.candidate_model = make_model(self.n_of_products, self.n_of_customers)
        self.load_weight_and_compile(load_checkpoint)

        # model artifacts
        self.history = None
        self.product_embeddings, self.customer_embeddings = None, None

    def load_weight_and_compile(self, load_checkpoint):
        if load_checkpoint:
            load_weight(self.candidate_model, self.checkpoint_dir)
        # self.candidate_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae'])

    def train_candidate_model(self, epochs=100):
        print('checkpoint path', self.checkpoint_path)
        callbacks = []
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, save_freq='epoch',
                                                         save_weights_only=True)
        callbacks.append(cp_callback)
        history = self.candidate_model.fit([self.train_ds.product_id, self.train_ds.customer_id], self.train_ds.rating,
                                           batch_size=32,
                                           epochs=epochs,
                                           callbacks=callbacks
                                           )
        self.history = history

    def eval_candidate_model(self, load_checkpoint=True):
        self.load_weight_and_compile(load_checkpoint)
        self.candidate_model.evaluate((self.test_ds.product_id, self.test_ds.customer_id), self.test_ds.rating,
                                      batch_size=1)

    def _set_embeddings(self):
        self.customer_embeddings = self.candidate_model.get_layer('customer_embeddings').get_weights()[
            0]  # shape: (n_of_customers_embedding_dim)
        self.product_embeddings = self.candidate_model.get_layer('product_embeddings').get_weights()[
            0]  # shape: (n_of_products, embedding_dim)
        # return np.array(self.customer_embeddings), np.array(self.product_embeddings)

    def recommend(self, customer_id, num_of_products):
        embedding_name = 'product_embeddings'
        self._set_embeddings()
        # Shape: (1, embedding_dim) x (embedding_dim, n_of_products) => (1, n_of_products)
        products = np.matmul(self.product_embeddings[customer_id], self.product_embeddings.T)
        recommended_product_idxes = np.argpartition(products, -num_of_products)[-num_of_products:]
        return recommended_product_idxes, products[recommended_product_idxes]

    def visualize_embedding(self):
        log_dir = 'logs/embeddings'
        embedding_name = 'customer_embeddings'
        make_dir(log_dir)
        self._set_embeddings()
        product_embeddings_tensor = tf.Variable(self.product_embeddings)
        customer_embeddings_tensor = tf.Variable(self.customer_embeddings)
        checkpoint = tf.train.Checkpoint(customer_embeddings=customer_embeddings_tensor)
        checkpoint.save(os.path.join(log_dir, '{}.ckpt'.format(embedding_name)))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = '{}/.ATTRIBUTES/VARIABLE_VALUE'.format(embedding_name)
        projector.visualize_embeddings(log_dir, config)
