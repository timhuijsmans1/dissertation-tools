import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from df_builder import load_file_paths, df_builder, df_loader
from datetime import datetime

def add_prevalence_change(df):
    # replace all 0 values with mean to avoid infinite increments
    df['union_prevalence_pos'] = df['union_prevalence_pos'].replace(0, df['union_prevalence_pos'].mean())
    df['emoticon_prevalence_pos'] = df['emoticon_prevalence_pos'].replace(0, df['emoticon_prevalence_pos'].mean())
    df['union_prevalence_pos_change'] = df['union_prevalence_pos'].pct_change(1)
    df['emoticon_prevalence_pos_change'] = df['emoticon_prevalence_pos'].pct_change(1)
    return df

def prevalence_return_correlations(total_df, ticker):
    df = total_df[total_df['ticker'] == ticker]
    df = df.dropna()
    df = add_prevalence_change(df)
    df = df.dropna()
    
    union_overnight = df['union_prevalence_pos'].corr(df['Overnight Return'])
    union_intra = df['union_prevalence_pos'].corr(df['Daily Return'])

    emoticon_overnight = df['emoticon_prevalence_pos'].corr(df['Overnight Return'])
    emoticon_intra = df['emoticon_prevalence_pos'].corr(df['Daily Return'])

    output_dict = {
        'union_overnight': union_overnight, 
        'union_daily': union_intra,
        'emoticon_overnight': emoticon_overnight,
        'emoticon_daily': emoticon_intra
    }

    return output_dict

def average_tweet_count(df, ticker):
    ticker_df = df[df['ticker'] == ticker]
    avg_tweet_count = ticker_df['tweet_count'].mean()
    return avg_tweet_count

def union_vs_emoticon(path_to_ticker_folders, prevalence_time, return_type):
    ticker_names = [folder for folder in os.listdir(path_to_ticker_folders) if folder[0] != '.']
    total_df = df_loader(prevalence_time)
    union_dict_key = 'union_' + return_type
    emoticon_dict_key = 'emoticon_' + return_type
    series_to_plot = []
    for ticker in ticker_names:
        correlations = prevalence_return_correlations(total_df, ticker)
        union_correlation = correlations[union_dict_key]
        emoticon_correlation = correlations[emoticon_dict_key]
        avg_tweet_count = average_tweet_count(total_df, ticker)
        series_to_plot.append((ticker, union_correlation, emoticon_correlation, avg_tweet_count))
    
    return series_to_plot

def split_correlation_comparisons(comparisons):
    """
    The function sorts the correlation comparisons
    by the pos/neg scenarios. The reason for executing this
    is the plotting of comparisons later on, as this helps
    with the x-axis
    """

    # format is given by union_emoticon
    pos_pos = []
    pos_neg = []
    neg_pos = []
    neg_neg = []
    for comparison in comparisons:
        union_pos = comparison[1] > 0
        union_neg = comparison[1] < 0
        emoticon_pos = comparison[2] > 0
        emoticon_neg = comparison[2] < 0

        if union_pos and emoticon_pos:
            pos_pos.append(comparison)
        if union_pos and emoticon_neg:
            pos_neg.append(comparison)
        if union_neg and emoticon_pos:
            neg_pos.append(comparison)
        if union_neg and emoticon_neg:
            neg_neg.append(comparison)

    return pos_pos, pos_neg, neg_pos, neg_neg

def filter_for_tweet_count_threshold(single_plot_comparisons, tweet_count_threshold):
    sign = tweet_count_threshold[0]
    threshold = int(tweet_count_threshold[1:])
    filtered_comparisons = []
    for comparison in single_plot_comparisons:
        if sign == '>':
            if comparison[3] > threshold:
                filtered_comparisons.append(comparison)
        if sign == '<':
            if comparison[3] < threshold:
                filtered_comparisons.append(comparison)
    return filtered_comparisons

def sort_comparisons(single_plot_comparisons, sortkey, tweet_count_threshold):
    if sortkey == 'absolute':
        single_plot_comparisons = sorted(single_plot_comparisons, key=lambda x: abs(x[1] - x[2]))

    if sortkey == 'ratio':
        single_plot_comparisons = sorted(single_plot_comparisons, key=lambda x: x[1] / x[2])

    if sortkey == 'highest union':
        single_plot_comparisons = sorted(single_plot_comparisons, key=lambda x: x[1])
    
    if sortkey == 'tweet count':
        single_plot_comparisons = sorted(single_plot_comparisons, key=lambda x: x[3])
    
    if tweet_count_threshold != None:
        single_plot_comparisons = filter_for_tweet_count_threshold(single_plot_comparisons, tweet_count_threshold)

    ticker_list = [x[0] for x in single_plot_comparisons]
    union_list = [x[1] for x in single_plot_comparisons]
    emoticon_list = [x[2] for x in single_plot_comparisons]
    tweet_count_list = ["   " + str(int(x[3])) for x in single_plot_comparisons]

    count = 0
    highest_tweet_count = 0
    for i in range(len(union_list)):
        if emoticon_list[i] > union_list[i]:
            count += 1
            tweet_count_int = int(tweet_count_list[i].strip(" "))
            if tweet_count_int > highest_tweet_count:
                highest_tweet_count = tweet_count_int
    print(count, highest_tweet_count, len(ticker_list))

    return ticker_list, union_list, emoticon_list, tweet_count_list
        
def convert(lst):
    return [ -i for i in lst ]

def tornado_plot_of_series(
    comparison_tuples, 
    union_corr_sign, 
    emoticon_corr_sign, 
    prevalence_time, 
    return_time, 
    tweet_count_threshold, 
    fig_save_path=None, 
    sortby='tweet count'
    ):
    ticker_list, union_list, emoticon_list, tweet_count_list = sort_comparisons(
                                                                comparison_tuples, 
                                                                sortby, 
                                                                tweet_count_threshold
    )
    fig = go.Figure()
    if union_corr_sign == 'pos' and emoticon_corr_sign == 'pos':
        if tweet_count_threshold == None:
            yticksize = 5
        elif int(tweet_count_threshold[1:]) > 50:
            print("set to 10")
            yticksize = 15
        else:
            yticksize = 5
        
        fig.add_trace(go.Bar(y=ticker_list, x=union_list,
                        base=0,
                        marker_color='rgb(158,202,225)',
                        name='Union method<br>(Pearson Correlation)',
                        marker_line_color='rgb(8,48,107)',
                        orientation='h',
                        marker_line_width=1.5,
                        opacity= 0.7,
                        text=tweet_count_list,
                        textposition='outside'
        ))
        fig.add_trace(go.Bar(y=ticker_list, x=emoticon_list,
                        base=convert(emoticon_list),
                        marker_color='crimson',
                        name='Emoticon method<br>(Pearson Correlation)',
                        marker_line_color='red',
                        orientation='h',
                        marker_line_width=1.5,
                        opacity= 0.7,
        ))
    if union_corr_sign == 'pos' and emoticon_corr_sign == 'neg':
        yticksize = 10
        fig.add_trace(go.Bar(y=ticker_list, x=union_list,
                        base=0,
                        marker_color='rgb(158,202,225)',
                        name='Union method<br>(Pearson Correlation)',
                        marker_line_color='rgb(8,48,107)',
                        orientation='h',
                        marker_line_width=1.5,
                        opacity= 0.7,
                        text=tweet_count_list,
                        textposition='outside'
        ))
        fig.add_trace(go.Bar(y=ticker_list, x=convert(emoticon_list),
                        base=emoticon_list,
                        marker_color='crimson',
                        name='Emoticon method<br>(Pearson Correlation)',
                        marker_line_color='red',
                        orientation='h',
                        marker_line_width=1.5,
                        opacity= 0.7,
        ))
    if union_corr_sign == 'neg' and emoticon_corr_sign == 'pos':
        yticksize = 10
        fig.add_trace(go.Bar(y=ticker_list, x=convert(union_list),
                        base=union_list,
                        marker_color='rgb(158,202,225)',
                        name='Union method<br>(Pearson Correlation)',
                        marker_line_color='rgb(8,48,107)',
                        orientation='h',
                        marker_line_width=1.5,
                        opacity= 0.7,
                        text=tweet_count_list,
                        textposition='outside'
        ))
        fig.add_trace(go.Bar(y=ticker_list, x=emoticon_list,
                        base=0,
                        marker_color='crimson',
                        name='Emoticon method<br>(Pearson Correlation)',
                        marker_line_color='red',
                        orientation='h',
                        marker_line_width=1.5,
                        opacity= 0.7,
        ))
    fig.update_layout(
        height=500,
        margin=dict(t=50,l=10,b=10,r=10),
    # title_text=f"{prevalence_time}-market positive sentiment & {return_time} return",
    # title_font_family="sans-serif",
    # #legend_title_text=’Financials’,
    # title_font_size = 25,
    # title_font_color="darkblue",
    # title_x=0.5 #to adjust the position along x-axis of the title
    )
    fig.update_layout(
            barmode='overlay',
            xaxis_tickangle=-45,
            xaxis_range=[-0.53, 0.53],
            xaxis_tickfont_size=20,
            legend=dict(
                        x=0.70,
                        y=0.01,
                        bgcolor='rgba(255, 255, 255, 0)',
                        bordercolor='rgba(255, 255, 255, 0)',
                        # font=dict(
                        #     size=15
                        # )
            ),
            yaxis=dict(
                title='Ticker',
                titlefont_size=16,
                tickfont_size=yticksize
            ),
            bargap=0.30,
    )
            
    if fig_save_path:
        if tweet_count_threshold == None:
            tweet_count_threshold = 'all_tweets'
        file_name = f'{union_corr_sign}_{emoticon_corr_sign}_{prevalence_time}_{return_time}_{tweet_count_threshold}.png'
        fig.write_image(os.path.join(fig_save_path, file_name), scale=6)

def tornado_plot(
    data_folder, 
    tornado_folder_path, 
    union_sign, 
    emoticon_sign, 
    prevalence_time, 
    return_time,
    tweet_count_threshold,
    ):
    # comparisons for intra/intra including all pos/neg scenarios
    correlation_sign = union_sign + emoticon_sign
    all_correlation_comparisons = union_vs_emoticon(data_folder, prevalence_time, return_time)
    pos_pos, pos_neg, neg_pos, neg_neg = split_correlation_comparisons(all_correlation_comparisons)
    correlation_sign_dict = {'pospos': pos_pos, 'posneg': pos_neg, 'negpos': neg_pos, 'negneg': neg_neg}
    tornado_plot_of_series(
        correlation_sign_dict[correlation_sign], 
        union_sign, 
        emoticon_sign, 
        prevalence_time, 
        return_time, 
        tweet_count_threshold,
        tornado_folder_path, 
        'highest union',
    )

if __name__ == "__main__":
    TORNADO_FOLDER = '../images/tornado_plots/individual'
    DATA_FOLDER = "../../labelled_data2predictions/data/pipeline_output"
    UNION_SIGN = input("union sign (pos/neg): ")
    if UNION_SIGN != 'pos' and UNION_SIGN != 'neg':
        UNION_SIGN = input("TRY AGAIN (pos/neg): ")
    EMOTICON_SIGN = input("emoticon sign (pos/neg): ")
    if EMOTICON_SIGN != 'pos' and EMOTICON_SIGN != 'neg':
        EMOTICON_SIGN = input("TRY AGAIN (pos/neg): ")
    PREVALENCE_TIME = input("prevalence time (Pre/Intra): ")
    if PREVALENCE_TIME != 'Pre' and PREVALENCE_TIME != 'Intra':
        PREVALENCE_TIME = input("TRY AGAIN (Pre/Intra): ")
    RETURN_TIME = input("return time (overnight/daily): ")
    if RETURN_TIME != 'overnight' and RETURN_TIME != 'daily':
        RETURN_TIME = input("TRY AGAIN (overnight/daily): ")
    TWEET_COUNT_THRESHOLD = input("tweet count threshold (None, >/<integer): ")
    if TWEET_COUNT_THRESHOLD == 'None' or TWEET_COUNT_THRESHOLD == 'none':
        TWEET_COUNT_THRESHOLD = None
    tornado_plot(DATA_FOLDER, TORNADO_FOLDER, UNION_SIGN, EMOTICON_SIGN, PREVALENCE_TIME, RETURN_TIME, TWEET_COUNT_THRESHOLD)
