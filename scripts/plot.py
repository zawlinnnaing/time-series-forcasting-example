import pandas as pd
import arrow
from matplotlib import pyplot as plt
from scripts.constants import ARROW_DATE_TIME_FORMAT, MATRICES

df = pd.read_csv('../output/data.csv')

date_time = [arrow.get(date_time_str, ARROW_DATE_TIME_FORMAT) for date_time_str in df['Date Time'].values]

plot_cols = MATRICES[0:4]

plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)

plt.show()
