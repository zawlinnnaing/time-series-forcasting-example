import pandas as pd
import uuid
import numpy as np
import random
import os
from tools.file_utils import make_dir

PRODUCTS_COUNT = 10000
CUSTOMERS_COUNT = 500
EMBEDDING_DIMENSION = 300
N_OF_PRODUCT_TYPES = 100

SHOPS_COUNT = 50

OUTPUT_DIR = '../output'

make_dir(OUTPUT_DIR)

TYPES_ARRAY = [
    "\nfiction ",
    "123123",
    "Laptops",
    "a tes type for default variant",
    "a test type for null variant",
    "a test vendor",
    "aaa",
    "adult elt\n\n",
    "agriculture",
    "archaeology",
    "architecture",
    "art",
    "art &  science",
    "art &  science\n",
    "art &  science\n\n",
    "art & science",
    "art & science\n\n",
    "as & a level",
    "as & a level\n",
    "asdfsadf",
    "asian history|\n",
    "autobiographies, biographies and memoirs",
    "autobiography",
    "beauty tools",
    "biochemistry",
    "biographies",
    "biography",
    "biography\n\n",
    "biology",
    "body mask",
    "book",
    "brand new",
    "buddhist literature",
    "buddhist sculpture",
    "burma--history--periodicals",
    "business",
    "business & economic",
    "business & economics",
    "business & economics\n\n",
    "business & economics\n ",
    "cabinet",
    "cat accessories ",
    "cat food",
    "chair",
    "chemical free vegetables",
    "chemistry",
    "children activities",
    "children hobbies",
    "childrens dictionary",
    "classic",
    "cloth",
    "clothed",
    "clothing",
    "comics",
    "compurer science",
    "computer",
    "computer\n\n",
    "computer\n\n ",
    "computer science",
    "cooking",
    "crime fiction",
    "cupboard",
    "decoration and ornament--burma--themes, motives.",
    "default",
    "design",
    "dish",
    "dog food",
    "dog supplements ",
    "drawing book",
    "drinks",
    "drone",
    "early child hood reader",
    "early childhood",
    "early childhood\n\n",
    "economic",
    "education",
    "education material",
    "educational material",
    "eggs",
    "elt",
    "elt  supplementary",
    "elt reference",
    "elt/ essays and letters",
    "encyclopedia",
    "engineering",
    "engineering\n\n",
    "engineering and scientists",
    "environment",
    "estest",
    "ethnology",
    "eye",
    "face",
    "fiction",
    "fiction\n\n",
    "fiction \n",
    "fition",
    "flower",
    "food",
    "forestry, biotechnology and genetics",
    "fruits",
    "furniture",
    "gems",
    "general knowledge",
    "general knowledge\n",
    "general knowledge\n\n",
    "genetic",
    "geography",
    "geology",
    "government and politics",
    "graphic novel",
    "graphic tshirt",
    "groceries",
    "guard rail",
    "history",
    "history\n",
    "history & culture",
    "history and political\n",
    "history culture",
    "history of architecture",
    "ielts",
    "igcse",
    "igcse as & a level",
    "international relation",
    "japan combine v-belt",
    "japan combine vbelt",
    "jcbelt",
    "jp combine vbelt",
    "kids book",
    "language",
    "laptop",
    "law",
    "lip",
    "lipstick",
    "literature",
    "lockers",
    "lotion",
    "management",
    "management\n\n",
    "marketing & management",
    "marketing & management\n\n\n",
    "mathematics",
    "medical",
    "medical\n",
    "men's t-shirt",
    "mgp products",
    "moisturizer",
    "mountaineering",
    "myanmar women",
    "newborn",
    "non ficition",
    "non-fiction",
    "nonfiction",
    "novel",
    "novel & story",
    "nursing",
    "parents",
    "pharmacy",
    "philosophy",
    "physics",
    "physiology",
    "political",
    "politics",
    "power tools",
    "pre kg",
    "pre-school book",
    "primary  reader",
    "primary 1",
    "primary 2",
    "primary 5",
    "primary 6",
    "primary computer science",
    "primary conputer science",
    "primary course book",
    "primary course book\n\n",
    "primary elt",
    "primary enviromental",
    "primary graded readers",
    "primary phonics",
    "primary phonics reader",
    "primary readaer",
    "primary reader",
    "primary soccial studies",
    "primary supplementary",
    "promotion set",
    "psychology",
    "psychology \n",
    "reader",
    "reader ",
    "reference",
    "reference \n",
    "reference \n\n",
    "reference/ test",
    "reference/test",
    "references",
    "religion",
    "safe box",
    "science",
    "scrub",
    "secondary",
    "secondary course book",
    "self-development",
    "self-help",
    "self-help book",
    "self-improvement",
    "self-improvement\n",
    "self-improvement\n\n",
    "shirt",
    "skin care",
    "social science",
    "sociology",
    "sock",
    "soup",
    "sss",
    "stationary",
    "statistics",
    "sticker books",
    "storybooks",
    "supplementry",
    "sushi",
    "t-shirt",
    "table",
    "test book",
    "toys",
    "travel",
    "travel\n\n",
    "v-belt",
    "vegetables",
    "wallpaper",
    "wearable",
    "yaung adult reader",
    "young  adult  reader",
    "young adault reader",
    "young adualt reader",
    "young adult reader",
    "young adult readers",
    "ကလေးစာပေ",
    "ကာတွန်း",
    "စိုက်ပျိုးရေး စာအုပ်",
    "ထမင်းဟင်းတခါပြင်",
    "နိုင်ငံရေး",
    "နိုင်ငံရေး စာအုပ်",
    "နံနက္စာ",
    "ပညာရေး စာအုပ်",
    "ပန်း အမျိုးမျိုး",
    "ပိန္နွဲ သီး",
    "ပြောင်ကြိုး",
    "ဗိသုကာပညာ",
    "ဗိသုကာပညာ စာအုပ်",
    "ဘုရားပန်း အမျိုးမျိုး",
    "မနက်စာ",
    "မွေးမြူရေး",
    "ရိတ်ခြွေကြိုး",
    "ရိတ်ခြွေကြိုး (ဘီပြောင်ကြိုး)",
    "ရိတ်ခြွေချိန်းကြိုး",
    "သမိုင်းစာအုပ်",
    "သိပ္ပံပညာ စာအုပ်",
    "သိသာ (အားတိုးဆေး)",
    "ဟင်း",
    "ဟင်းရွက်",
    "ဟင်းသီးဟင်းရွက် အမျိုးမျိုး",
    "အတ္ထုပ္ပတ္တိ",
    "အမျိုးသားအားကစားအဝတ်အထည်များ",
    "ေပ"
]


def generate_uuid():
    return uuid.uuid4()


def generate_rating():
    return random.randrange(1, 6)


def generate_rating_df(customers_count):
    # Out of simplicity, assume that customer can buy products from any shop.
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

    for customer_idx in range(0, customers_count):
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


def generate_products_df():
    result = list()
    for i in range(0, PRODUCTS_COUNT):
        result.append({
            'index': i,
            'product_id': generate_uuid(),
            'type': random.choice(TYPES_ARRAY),
            'vendor_idx': random.randrange(0, N_OF_PRODUCT_TYPES),
            'p_shop_idx': random.randrange(0, SHOPS_COUNT, 1)
        })
    return pd.DataFrame(result)


def generate_customers_df():
    result = list()
    for i in range(0, CUSTOMERS_COUNT):
        result.append({
            'index': i,
            'customer_id': generate_uuid(),
            'c_shop_idx': random.randrange(0, SHOPS_COUNT, 1),
            'gender': random.randrange(0, 2)
        })
    return pd.DataFrame(result)


shops_vector = generate_ids_vector(SHOPS_COUNT)

rating_df = generate_rating_df(CUSTOMERS_COUNT)
products_df = generate_products_df()
customers_df = generate_customers_df()

output_path = os.path.join(OUTPUT_DIR, 'rating.csv')
rating_df.to_csv(output_path)
products_df.to_csv(os.path.join(OUTPUT_DIR, 'products.csv'))
customers_df.to_csv(os.path.join(OUTPUT_DIR, 'customers.csv'))
