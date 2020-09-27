import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import config as cfg
from tools.utils import filter_index_and_null


class DataProcessor:
    """
    Time series data processor
    """

    def __init__(self, data_dir=cfg.DATA_DIR):
        self.data_dir = data_dir
        self.split_ratio = cfg.SPLIT_RATIO
        self.df = pd.read_csv(self.data_dir)
        self.df.pop('Date Time')
        self.df_len = len(self.df)

        # Split the dataset
        self.train_df = self.df[0:int(self.split_ratio['train'] * self.df_len)]
        val_end = self.split_ratio['train'] + self.split_ratio['val']
        self.val_df = self.df[int(self.split_ratio['train'] * self.df_len): int(self.df_len * val_end)]
        test_start = 1 - self.split_ratio['test']
        self.test_df = self.df[int(self.df_len * test_start):]

        # normalize dataset using only train_mean and train_std to prevent model from accessing values in val and
        # test sets.
        train_mean, train_std = self.train_df.mean(), self.train_df.std()
        self.train_norm_df = (self.train_df - train_mean) / train_std
        self.val_norm_df = (self.val_df - train_mean) / train_std
        self.norm_df = (self.df - train_mean) / train_std
        self.test_norm_df = (self.test_df - train_mean) / train_std

    def show_info(self):
        print(self.train_df.describe())
        print(self.test_df.describe())


class RecommendationDataProcessor:
    def __init__(self, data_dir='output'):
        self.data_dir = data_dir

        # Read data
        products_path = os.path.join(data_dir, 'products.csv')
        customers_path = os.path.join(data_dir, 'customers.csv')
        ratings_path = os.path.join(data_dir, 'rating.csv')
        print("===> Reading csvs...")
        self.products_df = pd.read_csv(products_path)
        self.customers_df = pd.read_csv(customers_path)
        self.ratings_df = pd.read_csv(ratings_path)
        self.product_ids = self.products_df.pop('product_id')
        self.customer_ids = self.customers_df.pop('customer_id')

        self.products_df = self.transform_column_to_cat(self.products_df, 'type')

        self.product_columns = list(filter(filter_index_and_null, self.products_df.columns[1:]))
        self.customer_columns = list(filter(filter_index_and_null, self.customers_df.columns[1:]))
        self.product_feature_dim, self.customer_feature_dim = len(self.product_columns), len(self.customer_columns)
        self.feature_columns = self.product_columns + self.customer_columns
        self.features_depth = self._get_features_depth()

        # process data
        result_columns = self.product_columns + self.customer_columns + ['label']
        self._result_df = pd.DataFrame(columns=result_columns)

        print("===> Processing data...")

        with tqdm(total=len(self.ratings_df)) as p_bar:
            for idx, row in self.ratings_df.iterrows():
                product_row = (self.products_df.loc[row['product_id'], self.product_columns]).values
                customer_row = (self.customers_df.loc[row['customer_id'], self.customer_columns]).values
                result_row = list(product_row) + list(customer_row) + [row['rating']]
                result_dict = {}
                for index, result_item in enumerate(result_row):
                    result_dict[result_columns[index]] = result_item
                self._result_df = self._result_df.append(result_dict, ignore_index=True)
                p_bar.update(1)
        print("===> Processed Data.")
        # make dataset
        total_rows = len(self._result_df.index)
        train_len = int(0.7 * total_rows)
        labels = np.array(self._result_df.pop('label').values, dtype=np.float)
        labels = np.expand_dims(labels, axis=1)
        # products_df = self._result_df[self.product_columns]
        # customers_df = self._result_df[self.customer_columns]
        features = np.array(self._result_df.values, dtype=np.float)
        print('features shape', features.shape)
        # product_features, customers_features = np.array(products_df.values, dtype=np.float), np.array(
        #     customers_df.values, dtype=np.float)
        # Shapes: Feature (num_of_rows, num_of_features) , Label (num_of_rows, 1)
        self.ds, self.train_ds, self.test_ds = self._make_ds((features, labels), train_len, total_rows)
        # self.product_ds, self.train_product_ds, self.test_product_ds = self._make_np_ds(product_features,
        #                                                                                 train_len,
        #                                                                                 total_rows)
        # self.customer_ds, self.train_customer_ds, self.test_customer_ds = self._make_np_ds(customers_features,
        #                                                                                    train_len,
        #                                                                                    total_rows)
        # self.label_ds, self.train_label_ds, self.test_label_ds = self._make_np_ds(labels, train_len, total_rows)

    def _get_features_depth(self):
        features_depth = list()

        def _append_features_depth(col_name, df):
            features_depth.append(pd.unique(df.loc[:, col_name]).shape[0])

        for product_col in self.product_columns:
            _append_features_depth(product_col, self.products_df)
        for customer_col in self.customer_columns:
            _append_features_depth(customer_col, self.customers_df)
        return features_depth

    @staticmethod
    def _make_np_ds(features, train_len, total_rows):
        np.random.shuffle(features)
        test_len = total_rows - train_len
        return features, features[:train_len, :], features[-test_len:, :]

    def _make_ds(self, features, train_len, total_rows):
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.map(self._map_ds).shuffle(total_rows)
        train_dataset = dataset.take(train_len)
        test_dataset = dataset.skip(train_len)
        return dataset, train_dataset, test_dataset,

    def _map_ds(self, feature, label):
        product_feature = feature[:len(self.product_columns)]
        product_feature_len = len(product_feature)
        product_one_hot_vector = self._make_one_hot_encoding(product_feature, self.features_depth[:product_feature_len])
        customer_feature = feature[-len(self.customer_columns):]
        customer_feature_len = len(customer_feature)
        customer_one_hot_vector = self._make_one_hot_encoding(customer_feature,
                                                              self.features_depth[-customer_feature_len:])
        return {
                   'product_input': product_one_hot_vector,
                   'customer_input': customer_one_hot_vector,
               }, label
        # return  [product_feature, customer_feature]

    @staticmethod
    def _make_one_hot_encoding(feature, features_depth):
        one_hot_vectors_list = []
        for index in range(len(features_depth)):
            per_feature = tf.gather(feature, indices=[index], axis=0)
            one_hot_vector_per_feature = tf.one_hot(tf.cast(per_feature, dtype=tf.int32), features_depth[index])
            one_hot_vectors_list.append(one_hot_vector_per_feature)
        return tf.concat(one_hot_vectors_list, axis=-1)

    @staticmethod
    def transform_column_to_cat(df, col_name):
        df[col_name] = pd.Categorical(df[col_name])
        df[col_name] = df[col_name].cat.codes
        return df
