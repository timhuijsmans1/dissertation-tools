import json
import os
import numpy as np

from df_builder import df_builder, prevalence_return_correlations, load_file_paths

def line_compiler(union_vals, emoticon_vals):
    union_line = ' & '.join(['{:.3f}'.format(val) for val in union_vals])
    union_line = 'Union method & ' + union_line + ' \\\\'
    emoticon_line = ' & '.join(['{:.3f}'.format(val) for val in emoticon_vals])
    emoticon_line = 'Emoticon method & ' + emoticon_line + ' \\\\'
    
    two_lines = union_line + '\n' + emoticon_line

    return two_lines

def caption_label_compiler(ticker):
    caption_line = '\\caption{' + ticker + '}\n'
    label_line = '\\label{table:' + ticker + '_corr}'
    return caption_line + label_line

def table_compiler(data, caption, head_path, bottom1_path, bottom2_path):
    with open(head_path, 'r') as f:
        head = f.read()
    with open(bottom1_path, 'r') as f:
        bottom1 = f.read()
    with open(bottom2_path, 'r') as f:
        bottom2 = f.read()
    table = head + data + bottom1 + caption + bottom2

    return table

def coefficient_table_compiler(
    pre_correlations, 
    intra_correlations,
    ticker,
    head_path,
    bottom1_path,
    bottom2_path
    ):
    union_values = (
        pre_correlations['union_overnight'], 
        pre_correlations['union_intra'], 
        intra_correlations['union_overnight'], 
        intra_correlations['union_intra']
    )
    emoticon_values = (
        pre_correlations['emoticon_overnight'],
        pre_correlations['emoticon_intra'],
        intra_correlations['emoticon_overnight'],
        intra_correlations['emoticon_intra']
    )
    data_to_write = line_compiler(union_values, emoticon_values)
    caption_label = caption_label_compiler(ticker)
    table = table_compiler(
                data_to_write, 
                caption_label, 
                head_path, 
                bottom1_path, 
                bottom2_path
    )
    return table

def write_tables(
        daily_file_paths, 
        head_path,
        bottom1_path,
        bottom2_path,
        table_out_path
    ):
    """
    builds the total text file containing all correlation tables
    of the individual NASDAQ-100 tickers
    """
    total_pre_df = df_builder(daily_file_paths, 'Pre')
    total_intra_df = df_builder(daily_file_paths, 'Intra')
    with open(table_out_path, 'w') as f:
        for stock_ticker in total_pre_df['ticker'].unique():
            pre_correlations = prevalence_return_correlations(total_pre_df, stock_ticker) # this works with the pre-market positive sentiment prevalence
            intra_correlations = prevalence_return_correlations(total_intra_df, stock_ticker) # this works with the intra-market positive sentiment prevalences
            table = coefficient_table_compiler(
                pre_correlations,
                intra_correlations,
                stock_ticker,
                head_path,
                bottom1_path,
                bottom2_path
            )
            f.write(table + '\n')
    

if __name__ == "__main__":
    DATA_FOLDER = "../../labelled_data2predictions/data/pipeline_output"
    HEAD_PATH = '../correlation_tables/top.txt'
    BOTTOM1_PATH = '../correlation_tables/bottom1.txt'
    BOTTOM2_PATH = '../correlation_tables/bottom2.txt'
    daily_file_paths = load_file_paths(DATA_FOLDER)
    write_tables(daily_file_paths, HEAD_PATH, BOTTOM1_PATH, BOTTOM2_PATH, '../correlation_tables/correlation_tables.txt')