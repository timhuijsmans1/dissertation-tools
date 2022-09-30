import json
import os
import numpy as np

from df_builder import df_builder, prevalence_return_correlations

def single_file_path_loader(ticker_parent_dir_path, ticker):
    daily_file_paths = {}
    prediction_folder = os.path.join(ticker_parent_dir_path, ticker, "predictions")
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

    return daily_file_paths

def single_stock_correlations(daily_file_paths, stock_ticker):
    total_pre_df = df_builder(daily_file_paths, 'Pre')
    total_intra_df = df_builder(daily_file_paths, 'Intra')
    print('pre-market')
    print(prevalence_return_correlations(total_pre_df, stock_ticker)) # this works with the pre-market positive sentiment prevalence
    print('intra-market')
    print(prevalence_return_correlations(total_intra_df, stock_ticker)) # this works with the intra-market positive sentiment prevalences

if __name__ == "__main__":
    DATA_FOLDER = "../../labelled_data2predictions/data/pipeline_output"
    stock_ticker = input('stock ticker:')
    daily_file_paths = single_file_path_loader(DATA_FOLDER, stock_ticker)
    single_stock_correlations(daily_file_paths, stock_ticker)