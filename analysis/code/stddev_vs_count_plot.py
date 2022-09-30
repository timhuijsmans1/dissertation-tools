import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from df_builder import load_file_paths, df_builder, df_loader
from pearson_correlation_plots import average_tweet_count

from datetime import datetime

def data_series_builder(df):
    union_devs = []
    emoticon_devs = []
    avg_counts = []
    for ticker in df['ticker'].unique():
        avg_count = average_tweet_count(df, ticker)
        ticker_df = df[df['ticker'] == ticker]
        union_std_dev = ticker_df['union_prevalence_pos'].std()
        emoticon_std_dev = ticker_df['emoticon_prevalence_pos'].std()
        union_devs.append(union_std_dev)
        emoticon_devs.append(emoticon_std_dev)
        avg_counts.append(int(avg_count))
    
    return avg_counts, union_devs, emoticon_devs

def std_dev_plot(x, y1, y2):
    all_std_devs = set(y1) | set(y2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x, y1, s=10, c='orange', marker="s", label='Union model', alpha=0.6)
    ax1.scatter(x, y2, s=10, c='blue', marker="o", label='Emoticon model', alpha=0.6)
    ax1.vlines(x=60, ymin=0, ymax=max(all_std_devs), color='green', linewidth=1, label='average count = 50')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('Positive prevalence standard deviation', fontsize=13)
    plt.xlabel('Average Tweet count', fontsize=13)

    plt.legend(loc='upper right', fontsize='large');
    plt.savefig(f'../images/std_devs.png', dpi=300)
    

if __name__ == "__main__":
    df = df_loader('Pre')
    counts, union_dev, emoticon_dev = data_series_builder(df)
    std_dev_plot(counts, union_dev, emoticon_dev)
