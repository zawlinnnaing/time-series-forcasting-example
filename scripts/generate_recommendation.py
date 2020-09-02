import pandas as pd
import uuid
import numpy as np
import random
import os
from tools.file_utils import make_dir

PRODUCTS_COUNT = 10000
CUSTOMERS_COUNT = 500
EMBEDDING_DIMENSION = 300

SHOPS_COUNT = 50

OUTPUT_DIR = '../output'

make_dir(OUTPUT_DIR)


def generate_uuid():
    return uuid.uuid4()


def generate_rating():
    return random.randrange(1, 6)


def generate_rating_df(customers_vector, products_vector):
    rating_length = random.randrange(4500, 5500)
    initial_idx = 0
    result_dimension = (rating_length, 3)
    result_dict = {
        'customer_id': [],
        'product_id': [],
        'rating': []
    }

    def get_random_products():
        return np.random.choice(range(0, PRODUCTS_COUNT), random.randrange(7, 50, 1))

    for (customer_idx, customer) in enumerate(customers_vector):
        random_products = get_random_products()
        for product_idx in random_products:
            result_dict['product_id'].append(product_idx)
            result_dict['customer_id'].append(customer_idx)
            result_dict['rating'].append(generate_rating())
    return pd.DataFrame(result_dict)


def generate_random_vector(dimension):
    """
    :param dimension: vector dimension
    :return: Shape(dimension)
    """
    return np.array([random.randrange(0, 500, 1) for i in range(0, dimension, 1)])


def generate_ids_vector(count):
    return np.array([generate_uuid() for i in range(0, count, 1)])



shops_vector = generate_ids_vector(SHOPS_COUNT)
products_ids_vector = generate_ids_vector(PRODUCTS_COUNT)
customer_ids_vector = generate_ids_vector(CUSTOMERS_COUNT)

rating_df = generate_rating_df(customer_ids_vector, products_ids_vector)
products_series = pd.Series(products_ids_vector, name='product_id')
customer_series = pd.Series(customer_ids_vector, name='customer_id')

output_path = os.path.join(OUTPUT_DIR, 'rating.csv')
rating_df.to_csv(output_path)
products_series.to_csv(os.path.join(OUTPUT_DIR, 'products.csv'))
customer_series.to_csv(os.path.join(OUTPUT_DIR, 'customers.csv'))
