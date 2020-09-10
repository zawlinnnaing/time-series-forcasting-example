from models.Recommendation import RecommendationModel
from models.recommendation_model import deep_nn_model
import argparse
import numpy as np
import os
import tensorflow as tf
from tools.data_processing import RecommendationDataProcessor

parser = argparse.ArgumentParser(description='Script for training recommendation model')

parser.add_argument('--train', action='store_true', help='Train the model')

parser.add_argument('--eval', action='store_true', help='Evaluate the model')

parser.add_argument('--data_dir', action='store', type=str, help="Data directory.", default='output')

parser.add_argument('--epoch', action='store', type=int, help="Number of epochs to train.", default=100)

if __name__ == '__main__':
    args = parser.parse_args()

    RATING_PATH = os.path.join(args.data_dir, 'rating.csv')
    PRODUCT_PATH = os.path.join(args.data_dir, 'products.csv')
    CUSTOMER_PATH = os.path.join(args.data_dir, 'customers.csv')

    # if not args.train or not args.eval:
    #     print('Need to specify "--train" or "--eval" flags or both.')
    #
    # recommendation_model = RecommendationModel(RATING_PATH, PRODUCT_PATH, CUSTOMER_PATH)
    # if args.train:
    #     recommendation_model.train_candidate_model(100)
    # if args.eval:
    #     recommendation_model.eval_candidate_model(True)
    #
    # recommended_product_idxes, recommended_product_values = recommendation_model.recommend(3, 5)

    dataProcessor = RecommendationDataProcessor(args.data_dir)
    # for features, label in dataProcessor.dataset.take(4):
    #     print("Features: {} \n, label: {} \n".format(features, label))

    model = deep_nn_model(dataProcessor.product_feature_dim, dataProcessor.customer_feature_dim)
    tf.keras.utils.plot_model(model, "dnn_recommendation_model_2.png")
    model.fit(dataProcessor.train_dataset, epochs=5, batch_size=32)
