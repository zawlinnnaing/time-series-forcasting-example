import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns


def plot_violin_plots(data_frame, x_label='Default violin plot', y_label='Value'):
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x=x_label, y=y_label, data=data_frame)
    plt.show()
