# Script for generating random samples for time series data.


import pandas as pd
import arrow
import numpy as np
import os
import random
from scripts.constants import ARROW_DATE_TIME_FORMAT, MATRICES
from tools.file_utils import make_dir

START_DATE_STR = '2017-01-01 00:00:00'

END_DATE_STR = '2020-01-01 00:00:00'

start_date_obj = arrow.get(START_DATE_STR, ARROW_DATE_TIME_FORMAT)
end_date_obj = arrow.get(END_DATE_STR, ARROW_DATE_TIME_FORMAT)

diff_date = end_date_obj - start_date_obj

diff_hours = diff_date.total_seconds() / 3600

OUTPUT_DIR = '../output'

OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'data.csv')

make_dir(OUTPUT_DIR)

obj = {}

# Just random pattern for every 50 time steps
TIME_STEPS = 50
time_steps_range = np.arange(50)
random_pattern = np.where(time_steps_range < 10, time_steps_range ** 3, (time_steps_range - 9) ** 2)


# add seasonality to dataset according to input_hour
def add_seasonality(input_hour):
    # Add non-stationary property to data
    input_hour = 0 if random.choice([0, 1, 2]) == 0 else input_hour
    # 50 is equal to time steps declared above
    return random_pattern[input_hour % TIME_STEPS] + random.randrange(0, 500, 1) + input_hour  # adding trend


if __name__ == '__main__':
    # hydrate object with keys
    for metric in MATRICES:
        obj[metric] = []

    # hydrate object with data
    for hour in range(int(diff_hours)):
        for metric in MATRICES:
            if metric == 'Date Time':
                result_date = start_date_obj.shift(hours=hour)
                obj[metric].append(result_date.format(ARROW_DATE_TIME_FORMAT))
            else:
                obj[metric].append(add_seasonality(hour))

    df = pd.DataFrame(data=obj, columns=MATRICES)
    df.to_csv(OUTPUT_PATH, index=False)
