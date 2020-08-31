import pandas as pd
from tools.file_utils import check_is_csv, make_dir
from sklearn.model_selection import train_test_split
from models.matrix_factorization import make_model
import os
import tensorflow as tf


class RecommendationModel:
    def __init__(self,
                 rating_path,
                 product_path,
                 customer_path,
                 checkpoint_dir='checkpoints/recommendation'):
        if (not check_is_csv(rating_path)) or (not check_is_csv(product_path) or (not check_is_csv(customer_path))):
            raise Exception('Input path {}, {}, {} is not csv'.format(rating_path, product_path, customer_path))
        self.rating_df = pd.read_csv(rating_path)
        self.product_series = pd.read_csv(product_path)
        self.customer_series = pd.read_csv(customer_path)

        # Create checkpoint dir
        self.checkpoint_dir = checkpoint_dir
        make_dir(checkpoint_dir)
        self.n_of_products, self.n_of_customers = self.product_series.size, self.customer_series.size
        self.train_ds, self.test_ds = train_test_split(self.rating_df)
        self.candidate_model = make_model(self.n_of_products, self.n_of_customers)

    def train_candidate_model(self, epochs=100, load_checkpoint=False):
        model_name = 'recommendation_candidate_model.ckpt'
        checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        callbacks = []
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_freq=100)
        callbacks.append(cp_callback)
        if load_checkpoint is True:
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            self.candidate_model.load_weights(latest_checkpoint)
            print('Checkpoint: {} loaded.'.format(latest_checkpoint))
        self.candidate_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
        history = self.candidate_model.fit([self.train_ds.product_id, self.train_ds.customer_id], self.train_ds.rating,
                                           batch_size=32,
                                           epochs=epochs,
                                           )
        return history
