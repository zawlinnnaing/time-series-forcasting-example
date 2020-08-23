from tools.data_processing import DataProcessor
from tools.window_generator import WindowGenerator
from tools.visualization import plot_violin_plots
from models.rnn import FeedBack
from models import compile_and_fit
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    OUTPUT_STEPS = 24
    NUM_FEATURES = 4
    data_processor = DataProcessor()
    data_window = WindowGenerator(
        train_df=data_processor.train_norm_df,
        test_df=data_processor.test_norm_df,
        val_df=data_processor.val_norm_df,
        input_width=24,
        label_width=OUTPUT_STEPS,
        shift=24)

    print(len(data_processor.val_norm_df))

    feedback_model = FeedBack(32, OUTPUT_STEPS, NUM_FEATURES)
    compile_and_fit(feedback_model, data_window)

    # example_data = tf.stack([
    #     np.array(data_processor.train_norm_df[:window_generator.total_window_size]),
    #     np.array(data_processor.train_norm_df[100:100 + window_generator.total_window_size]),
    # ])
    #
    # example_window = window_generator.split_window(example_data)
    #
    # print("Input windows shape", example_window[0].shape)
    # print("Output windows shape", lstm_model(example_window[0]).shape)

    # df_melt = data_processor.norm_df.melt(var_name='Column', value_name="Values")
    # print(df_melt)
