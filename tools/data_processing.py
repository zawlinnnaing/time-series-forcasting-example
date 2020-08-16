import pandas as pd
import tensorflow as tf

import config as cfg


class DataProcessor:
    def __init__(self, data_dir=cfg.DATA_DIR):
        self.data_dir = data_dir
        self.split_ratio = cfg.SPLIT_RATIO
        self.df = pd.read_csv(self.data_dir)
        self.df_len = len(self.df)

        self.train_df = self.df[0:int(self.split_ratio['train'] * self.df_len)]
        val_end = self.split_ratio['train'] + self.split_ratio['val']
        self.val_df = self.df[int(self.split_ratio['train'] + self.df_len): int(self.df_len * val_end)]
        test_start = 1 - self.split_ratio['test']
        self.test_df = self.df[int(self.df_len * test_start):]

