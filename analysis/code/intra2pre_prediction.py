import json
import os

import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from df_builder import df_builder, load_file_paths
from pathlib import Path
from datetime import datetime

def ratio_difference(new, old):
    return np.divide(
        np.subtract(new, old), old
    )

def calculate_intra2pre_change(sentiment_df):
    union_pre = sentiment_df['union_prevalence_pos_pre'].to_numpy()[1:]
    union_intra = sentiment_df['union_prevalence_pos_intra'].to_numpy()[:-1]
    emoticon_pre = sentiment_df['emoticon_prevalence_pos_pre'].to_numpy()[1:]
    emoticon_intra = sentiment_df['emoticon_prevalence_pos_intra'].to_numpy()[:-1]

    union_change = ratio_difference(union_pre, union_intra)
    emoticon_change = ratio_difference(emoticon_pre, emoticon_intra)

    sentiment_df['union_change_intra'] = sentiment_df['union_prevalence_pos_intra'].pct_change(1)
    sentiment_df['emoticon_change_intra'] = sentiment_df['emoticon_prevalence_pos_intra'].pct_change(1)

    # pad with an nan to line up yesterday intra to today pre
    # with todays stock return
    union_change = np.insert(union_change, 0, np.nan)
    emoticon_change = np.insert(emoticon_change, 0, np.nan)
    
    sentiment_df['union_change'] = union_change
    sentiment_df['emoticon_change'] = emoticon_change

    sentiment_df.dropna(inplace=True)

    return sentiment_df

def combine_pre_and_intra_sentiment(intra_sentiment_df, pre_sentiment_df):
    intra_sentiment_df.loc[:, 'union_prevalence_pos_pre'] = pre_sentiment_df['union_prevalence_pos']
    intra_sentiment_df.loc[:, 'emoticon_prevalence_pos_pre'] = pre_sentiment_df['emoticon_prevalence_pos']
    intra_sentiment_df.rename(columns={'union_prevalence_pos': 'union_prevalence_pos_intra',
                                       'emoticon_prevalence_pos': 'emoticon_prevalence_pos_intra'}, inplace=True)
    print(intra_sentiment_df.columns)
    return intra_sentiment_df

def correlations(df):
    union_correlation = df['union_change'].corr(df['Daily Return'])
    emoticon_correlation = df['emoticon_change'].corr(df['Daily Return'])
    union_pre_correlation = df['union_prevalence_pos_pre'].corr(df['Daily Return'])
    emoticon_pre_correlation = df['emoticon_prevalence_pos_pre'].corr(df['Daily Return'])
    union_change_intra = df['union_change_intra'].corr(df['Daily Return'])
    emoticon_change_intra = df['emoticon_change_intra'].corr(df['Daily Return'])

    output_dict = {
            'union correlation': union_change_intra,
            'emoticon correlation': emoticon_change_intra
    }
    return output_dict

def single_ticker_correlations(intra_sentiment_df, pre_sentiment_df, ticker):
    ticker_intra_df = intra_sentiment_df[intra_sentiment_df['ticker'] == ticker]
    ticker_pre_df = pre_sentiment_df[pre_sentiment_df['ticker'] == ticker]
    combined_df = combine_pre_and_intra_sentiment(
                                    ticker_intra_df, 
                                    ticker_pre_df
    )
    df_with_changes = calculate_intra2pre_change(combined_df)
    return correlations(df_with_changes), df_with_changes

def main(file_paths):
    pre_sentiment_df = df_builder(file_paths, 'Pre')
    intra_sentiment_df = df_builder(file_paths, 'Intra')
    corrs, final_df = single_ticker_correlations(intra_sentiment_df, pre_sentiment_df, 'AAPL')
    print(corrs)
    print(final_df['union_prevalence_pos_pre'].to_numpy())
    print(final_df['union_prevalence_pos_intra'].to_numpy())
    print(final_df['Daily Return'].to_numpy())

if __name__ == "__main__":
    DATA_FOLDER = "../../labelled_data2predictions/data/pipeline_output"
    DAILY_FILE_PATHS = load_file_paths(DATA_FOLDER)
    main(DAILY_FILE_PATHS)