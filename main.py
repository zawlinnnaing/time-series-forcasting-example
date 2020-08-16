from tools.data_processing import DataProcessor
from tools.window_generator import WindowGenerator
from tools.visualization import plot_violin_plots
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    data_processor = DataProcessor()
    window_generator = WindowGenerator(
        train_df=data_processor.train_norm_df,
        test_df=data_processor.test_norm_df,
        val_df=data_processor.val_norm_df,
        input_width=24,
        label_width=24,
        label_columns=['clicks', 'views'],
        shift=24)
    # window_generator.show_window_info()

    example_window = tf.stack([
        np.array(data_processor.train_norm_df[:window_generator.total_window_size]),
        np.array(data_processor.train_norm_df[100:100 + window_generator.total_window_size]),
    ])

    example_inputs, example_labels = window_generator.split_window(example_window)

    window_generator.example = example_inputs, example_labels

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'labels shape: {example_labels.shape}')

    # df_melt = data_processor.norm_df.melt(var_name='Column', value_name="Values")
    # print(df_melt)
