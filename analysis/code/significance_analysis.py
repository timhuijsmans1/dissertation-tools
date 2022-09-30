import os
import numpy as np

from scipy import stats
from df_builder import df_builder, load_file_paths, prevalence_return_correlations
from pearson_correlation_plots import average_tweet_count

def filter_all_correlations(correlations, tweet_count_threshold):
    sign = tweet_count_threshold[0]
    threshold = int(tweet_count_threshold[1:])
    if sign == '>':
        correlations = [correlation for correlation in correlations if correlation[2] > threshold]
    if sign == '<':
        correlations = [correlation for correlation in correlations if correlation[2] < threshold]
    return correlations

def union_vs_emoticon_correlations(
    path_to_ticker_folders, 
    ticker_names, 
    prevalence_time, 
    return_type, 
    tweet_count_threshold
    ):
    daily_file_paths = load_file_paths(path_to_ticker_folders)
    # building df of all tickers is actually redundant, but as of now
    # the load_file_paths loads all tickers internally
    # TODO: update load_file_paths to take in a list of tickers,
    # and update the scripts that use this function accordingly
    total_df = df_builder(daily_file_paths, prevalence_time)
    union_dict_key = 'union_' + return_type
    emoticon_dict_key = 'emoticon_' + return_type
    all_correlations = []
    for ticker in ticker_names:
        correlations = prevalence_return_correlations(total_df, ticker)
        union_correlation = correlations[union_dict_key]
        emoticon_correlation = correlations[emoticon_dict_key]
        avg_tweet_count = average_tweet_count(total_df, ticker)
        all_correlations.append((union_correlation, emoticon_correlation, avg_tweet_count))
    
    if tweet_count_threshold != None:
        all_correlations = filter_all_correlations(all_correlations, tweet_count_threshold)
        
    # split up into union and emoticon correlations
    union_correlations = [correlation[0] for correlation in all_correlations]
    emoticon_correlations = [correlation[1] for correlation in all_correlations]
    
    return union_correlations, emoticon_correlations

def load_ticker_names(path_to_ticker_folders):
    ticker_names = [folder for folder in os.listdir(path_to_ticker_folders) if folder[0] != '.']
    return ticker_names

def significance_analysis(union_correlations, emoticon_correlations):
    union_correlations = np.array(union_correlations)
    emoticon_correlations = np.array(emoticon_correlations)

    union_mean = np.mean(union_correlations)
    emoticon_mean = np.mean(emoticon_correlations)

    statistic, p_value = stats.ttest_ind(union_correlations, emoticon_correlations, equal_var=True)
    print(union_mean)
    print(emoticon_mean)
    print(p_value)

def all_statistics(ticker_list, ticker_folder):
    prevalence_times = ['Pre', 'Intra']
    return_types = ['intra', 'overnight']
    tweet_count_thresholds = [None, '<50', '>50']

    for prevalence_time in prevalence_times:
        for return_type in return_types:
            for tweet_count_threshold in tweet_count_thresholds:
                union_correlations, emoticon_correlations = union_vs_emoticon_correlations(
                    ticker_folder, 
                    ticker_list, 
                    prevalence_time, 
                    return_type, 
                    tweet_count_threshold
                    )
                print(prevalence_time, return_type, tweet_count_threshold)
                significance_analysis(union_correlations, emoticon_correlations)


if __name__ == "__main__":
    TICKER_FOLDER = "../../labelled_data2predictions/data/pipeline_output"
    ticker_list = load_ticker_names(TICKER_FOLDER)
    all_statistics(ticker_list, TICKER_FOLDER)