import json
import os

import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from df_builder import df_builder, load_file_paths, add_prevalence_change
from pathlib import Path
from datetime import datetime

def overnight_change_simulation(ticker_df):
    union_changes = ticker_df['union_prevalence_pos_change']
    emoticon_changes = ticker_df['emoticon_prevalence_pos_change']
    daily_returns = ticker_df['Daily Return']
    
    for i in range(len(union_changes)):
        union_returns = 0
        emoticon_returns = 0
        if union_changes[i] > 0:
            union_returns += 100000 * union_changes[i] * daily_returns[i]
            print(100000 * union_changes[i])
        if emoticon_changes[i] > 0:
            emoticon_returns += 100000 * emoticon_changes[i] * daily_returns[i]
        
    print(union_returns, emoticon_returns)

def add_changes_to_df(total_df, ticker):
    df = total_df[total_df['ticker'] == ticker]
    df = df.dropna()
    df = add_prevalence_change(df)
    df = df.dropna()
    return df

if __name__ == "__main__":
    DATA_FOLDER = "../../labelled_data2predictions/data/pipeline_output"
    DAILY_FILE_PATHS = load_file_paths(DATA_FOLDER)
    df = df_builder(DAILY_FILE_PATHS, 'Pre')
    apple_df = add_changes_to_df(df, "AAPL")
    overnight_change_simulation(apple_df)
