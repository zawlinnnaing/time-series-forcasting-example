import pandas as pd
import tensorflow as tf
import numpy as np
import config as cfg


class DataProcessor:
    def __init__(self, data_dir=cfg.DATA_DIR):
        self.data_dir = data_dir
        self.split_ratio = cfg.SPLIT_RATIO
        self.df = pd.read_csv(self.data_dir)
        self.df.pop('Date Time')
        self.df_len = len(self.df)

        # Split the dataset
        self.train_df = self.df[0:int(self.split_ratio['train'] * self.df_len)]
        val_end = self.split_ratio['train'] + self.split_ratio['val']
        self.val_df = self.df[int(self.split_ratio['train'] + self.df_len): int(self.df_len * val_end)]
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
