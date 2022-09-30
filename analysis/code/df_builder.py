import json
import os

import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime

def load_file_paths(pipeline_output_path):
    ticker_folders = [folder for folder in os.listdir(pipeline_output_path) if folder[0] != '.']
    daily_file_paths = {}
    for ticker in ticker_folders:
        prediction_folder = os.path.join(pipeline_output_path, ticker, "predictions")
        daily_folders = sorted([folder for folder in os.listdir(prediction_folder) if folder[0] != '.'])
        daily_file_paths[ticker] = {}
        daily_file_paths[ticker]['Pre'] = [
            os.path.join(prediction_folder, daily_folder, 'pre.json')
            for daily_folder in daily_folders
            if daily_folder[0] != '.'
        ]
        daily_file_paths[ticker]['Intra'] = [
            os.path.join(prediction_folder, daily_folder, 'intra.json') 
            for daily_folder in daily_folders
            if daily_folder[0] != '.']

    return(daily_file_paths)

def prevalence_extractor(file_path):
    with open(file_path) as f:
        data = json.load(f)
    emoticon_prevalence = data['emoticon']['prevalence']
    union_prevalence = data['union']['prevalence']
    tweet_count = len(data['union']['labels'])
    return union_prevalence, emoticon_prevalence, tweet_count

def prevalence_df_builder(all_file_paths, ticker, moment_of_day):
    all_prevalence = []
    for file_path in all_file_paths[ticker][moment_of_day]:
        union_prevalence, emoticon_prevalence, tweet_count = prevalence_extractor(file_path)
        date_string = os.path.dirname(file_path).split("/")[-1]
        date_object = datetime.strptime(date_string, '%Y-%m-%d')
        all_prevalence.append(
            (ticker,
            union_prevalence['negative'],
            union_prevalence['positive'], 
            emoticon_prevalence['negative'],
            emoticon_prevalence['positive'],
            tweet_count,
            date_object)
        )
    prevalence_df = pd.DataFrame(
                all_prevalence, 
                columns=[
                'ticker', 
                'union_prevalence_neg',
                'union_prevalence_pos', 
                'emoticon_prevalence_neg',
                'emoticon_prevalence_pos', 
                'tweet_count',
                'date'
                ]
    )
    return prevalence_df

def add_returns(ticker_df):
    # overnight returns
    close_prices = ticker_df['Close'].to_numpy()
    open_prices = ticker_df['Open'].to_numpy()

    realigned_close = close_prices[:-1]
    next_day_open_price = open_prices[1:]

    daily_return = np.divide(
        np.subtract(close_prices, open_prices), open_prices
    )
    overnight_return = np.divide(
        np.subtract(next_day_open_price, realigned_close), realigned_close
    )
    # pad the first row with a NaN, which leads to dropping the row
    # later on in the trading day dataframe
    overnight_return = np.insert(overnight_return, 0, np.nan)

    ticker_df['Daily Return'] = daily_return
    ticker_df['Overnight Return'] = overnight_return

    return ticker_df

def add_price_data(df, ticker):
    start_date = datetime.strftime(df['date'].min(), '%Y-%m-%d')
    end_date = datetime.strftime(df['date'].max(), '%Y-%m-%d')
    ticker_data = yf.download(ticker, start_date, end_date)
    ticker_df = add_returns(ticker_data)
    df.set_index('date', inplace=True, drop=False)
    df['Open'] = ticker_df['Open']
    df['Close'] = ticker_df['Close']
    df['Daily Return'] = ticker_df['Daily Return']
    df['Overnight Return'] = ticker_df['Overnight Return']

    return df

def df_builder(all_daily_file_paths, moment_of_day):
    ticker_list = list(all_daily_file_paths.keys())
    total_df = pd.DataFrame()
    for ticker in ticker_list:
        prevalence_df = prevalence_df_builder(all_daily_file_paths, ticker, moment_of_day)
        df_with_prices = add_price_data(prevalence_df, ticker)
        total_df = pd.concat([total_df, df_with_prices])
    return total_df

def add_prevalence_change(df):
    # replace all 0 values with mean to avoid infinite increments
    df['union_prevalence_pos'] = df['union_prevalence_pos'].replace(0, df['union_prevalence_pos'].mean())
    df['emoticon_prevalence_pos'] = df['emoticon_prevalence_pos'].replace(0, df['emoticon_prevalence_pos'].mean())
    df['union_prevalence_pos_change'] = df['union_prevalence_pos'].pct_change(1)
    df['emoticon_prevalence_pos_change'] = df['emoticon_prevalence_pos'].pct_change(1)
    return df

def add_random_prevalence(df):
    union_min_prevalence = df['union_prevalence_pos'].min()
    emoticon_min_prevalence = df['emoticon_prevalence_pos'].min()
    min_prevalence = min(union_min_prevalence, emoticon_min_prevalence)
    union_max_prevalence = df['union_prevalence_pos'].max()
    emoticon_max_prevalence = df['emoticon_prevalence_pos'].max()
    max_prevalence = max(union_max_prevalence, emoticon_max_prevalence)
    df['random_prevalence_pos'] = np.random.uniform(min_prevalence, max_prevalence, len(df))
    return df

def prevalence_return_correlations(total_df, ticker):
    df = total_df[total_df['ticker'] == ticker]
    df = df.dropna()
    df = add_prevalence_change(df)
    df = df.dropna()
    # df = add_random_prevalence(df)
    
    union_overnight = df['union_prevalence_pos'].corr(df['Overnight Return'])
    union_intra = df['union_prevalence_pos'].corr(df['Daily Return'])

    emoticon_overnight = df['emoticon_prevalence_pos'].corr(df['Overnight Return'])
    emoticon_intra = df['emoticon_prevalence_pos'].corr(df['Daily Return'])

    output_dict = {
        'union_overnight': union_overnight, 
        'union_intra': union_intra,
        'emoticon_overnight': emoticon_overnight,
        'emoticon_intra': emoticon_intra,

    }
    return output_dict

def df_writer(df, moment_of_prevalence):
    path = f'../dataframes/full_dataframe_{moment_of_prevalence}.csv'
    df.to_csv(path)

def df_loader(moment_of_prevalence):
    path = f'../dataframes/full_dataframe_{moment_of_prevalence}.csv'
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    DATA_FOLDER = "../../labelled_data2predictions/data/pipeline_output"
    DAILY_FILE_PATHS = load_file_paths(DATA_FOLDER)
    pre_df = df_builder(DAILY_FILE_PATHS, 'Pre')
    intra_df = df_builder(DAILY_FILE_PATHS, 'Intra')
    df_writer(pre_df, 'Pre')
    df_writer(intra_df, 'Intra')