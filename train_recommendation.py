import argparse
import os
import sys
from models.Recommendation import DeepNNRecommendationModel, MatrixRecommendationModel

parser = argparse.ArgumentParser(description='Script for training recommendation model')

parser.add_argument('--train', action='store_true', help='Train the model')

parser.add_argument('--model', choices=['dnn', 'matrix'], default='dnn')

parser.add_argument('--eval', action='store_true', help='Evaluate the model')

parser.add_argument('--data_dir', action='store', type=str, help="Data directory.", default='output')

parser.add_argument('--epoch', action='store', type=int, help="Number of epochs to train.", default=100)

parser.add_argument('--checkpoint_dir', action='store', type=str, help='Directory to store checkpoints',
                    default="checkpoints/")

if __name__ == '__main__':
    args = parser.parse_args()

    RATING_PATH = os.path.join(args.data_dir, 'rating.csv')
    PRODUCT_PATH = os.path.join(args.data_dir, 'products.csv')
    CUSTOMER_PATH = os.path.join(args.data_dir, 'customers.csv')

    if not args.train and not args.eval:
        print('Need to specify "--train" or "--eval" flags or both.')
        sys.exit()

    if args.model == 'matrix':
        recommendation_model = MatrixRecommendationModel(RATING_PATH, PRODUCT_PATH, CUSTOMER_PATH)
        if args.train:
            recommendation_model.train_candidate_model(args.epoch)
        if args.eval:
            recommendation_model.eval_candidate_model(True)

    elif args.model == 'dnn':
        recommendation_model = DeepNNRecommendationModel(args.data_dir, args.checkpoint_dir)
        if args.train:
            recommendation_model.train(epochs=args.epoch)
        if args.eval:
            recommendation_model.evaluate()
