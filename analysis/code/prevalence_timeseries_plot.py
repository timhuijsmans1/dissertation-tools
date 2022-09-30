import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from df_builder import load_file_paths, df_builder, df_loader
from pearson_correlation_plots import average_tweet_count

from datetime import datetime

def plot_multiple_cols(df):
    df['Date'] = np.array(df['date'].to_numpy())
    colums_to_plot = df.loc[
        :, 
        [
        'Date', 
        'Overnight Return', 
        'union_prevalence_pos', 
        'emoticon_prevalence_pos'
        ]
    ]
    colums_to_plot = colums_to_plot.rename(
        columns={
            'union_prevalence_pos': 'Union Method', 
            'emoticon_prevalence_pos': 'Emoticon Method'
        }
    )
    dfm = colums_to_plot.melt('Date', var_name='quantity', value_name='value')
    print(dfm.head())
    return dfm

def plot_prev_returns(df, moment_of_day):
    for stock_ticker in df['ticker'].unique():
        avg_tweet_count = average_tweet_count(df, stock_ticker)
        ticker_df = df[df['ticker'] == stock_ticker]
        ticker_df.dropna(inplace=True)
        melt_df = plot_multiple_cols(ticker_df)
        figure = plt.figure()
        figure.set_size_inches(20, 5)
        sns.set(style="whitegrid", rc={"lines.linewidth": 0.7})
        sns.catplot(x='Date', y="value", hue='quantity', data=melt_df, kind='point', s=1, legend=False)
        plt.title(f'Positive {moment_of_day}-market prevalence towards {stock_ticker}')
        plt.xticks(range(1, len(ticker_df), 30), rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        if stock_ticker == 'AAPL':   
            plt.legend(fontsize='large', title_fontsize='15')
        plt.tight_layout()
        plt.savefig(f'../images/all_timeseries/{moment_of_day}/{stock_ticker}_prevalence_returns.png', dpi=300)

if __name__ == "__main__":
    DATA_FOLDER = "../../labelled_data2predictions/data/pipeline_output"
    DAILY_FILE_PATHS = load_file_paths(DATA_FOLDER)
    # df = df_builder(DAILY_FILE_PATHS, 'Intra')
    # plot_prev_returns(df, 'Intra')
    df = df_loader('Pre')
    plot_prev_returns(df, 'Pre')
