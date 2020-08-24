from tools.data_processing import DataProcessor
from tools.window_generator import WindowGenerator
from models.rnn import FeedBack, single_shot_lstm
from models import compile_and_fit, load_weight
import os
import config as cfg

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

    # feedback_model = FeedBack(32, OUTPUT_STEPS, NUM_FEATURES)
    model = single_shot_lstm(OUTPUT_STEPS, NUM_FEATURES)
    compile_and_fit(model, data_window, 'lstm')
    checkpoint_dir = os.path.join(cfg.CHECKPOINT_PATH, 'lstm')
    model = load_weight(model, checkpoint_dir)

    data_window.plot('clicks', model)
