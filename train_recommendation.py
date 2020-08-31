from models.Recommendation import RecommendationModel

RATING_PATH = 'output/rating.csv'
PRODUCT_PATH = 'output/products.csv'
CUSTOMER_PATH = 'output/customers.csv'

if __name__ == '__main__':
    recommendation_model = RecommendationModel(RATING_PATH, PRODUCT_PATH, CUSTOMER_PATH)
    recommendation_model.train_candidate_model(100, True)
