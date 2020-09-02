from models.Recommendation import RecommendationModel
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Script for training recommendation model')

parser.add_argument('--train', action='store_true', help='Train the model')

parser.add_argument('--eval', action='store_true', help='Evaluate the model')


parser.add_argument('--epoch', action='store', type=int, help="Number of epochs to train.", default=100)

RATING_PATH = 'output/rating.csv'
PRODUCT_PATH = 'output/products.csv'
CUSTOMER_PATH = 'output/customers.csv'

if __name__ == '__main__':
    args = parser.parse_args()

    if not args.train or not args.eval:
        print('Need to specify "--train" or "--eval" flags or both.')

    recommendation_model = RecommendationModel(RATING_PATH, PRODUCT_PATH, CUSTOMER_PATH)
    if args.train:
        recommendation_model.train_candidate_model(100, True)
    if args.eval:
        recommendation_model.eval_candidate_model(True)

    recommendation_model.visualize_embedding()

    recommended_product_idxes, recommended_product_values = recommendation_model.recommend(3, 5)

    print('recommended_product_idxes', recommended_product_idxes, recommended_product_values)
