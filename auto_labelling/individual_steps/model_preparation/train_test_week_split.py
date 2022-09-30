import pandas as pd
import json
import os
from datetime import datetime, timedelta

def df_builder(data_path):
    tweets = []
    with open(data_path, 'r') as f:
        for line in f:
            tweet_data = json.loads(line)
            label = tweet_data['label']
            datetime_string = tweet_data['created_at']
            date_string = datetime_string.split('T')[0]
            date_object = datetime.strptime(date_string, '%Y-%m-%d')
            original_text = tweet_data['original_text']
            tokens = tweet_data['processed_tokens']
            emoticon_list = tweet_data['emoticon_list']
            cashtag_list = tweet_data['cashtag_list']
            n_grams = tweet_data['n-grams']
            engineered_features = [tweet_data['engineered features']]
            tweets.append((label, date_object, datetime_string, original_text, tokens, emoticon_list, cashtag_list, n_grams, engineered_features))
    labelled_tweet_df = pd.DataFrame(tweets, columns=['label', 'date', 'datetime string', 'original text', 'tokens', 'emoticon list', 'cashtag list', 'n-grams', 'engineered features'])
    start_date = labelled_tweet_df['date'].min()
    end_date = labelled_tweet_df['date'].max()
    return labelled_tweet_df, start_date, end_date

def test_path_compiler(df, base_path):
    # build filename
    start_date = df['date'].min()
    # write start date as string
    start_date_string = start_date.strftime('%Y-%m-%d') + '.dat'

    return os.path.join(base_path, start_date_string)

def write_df_to_engineered(df, path):
    with open(path, 'w') as f:
        for index, row in df.iterrows():
            dict_to_write = {}
            dict_to_write['label'] = row['label']
            dict_to_write['original_text'] = row['original text']
            dict_to_write['created_at'] = row['datetime string']
            dict_to_write['processed_tokens'] = row['tokens']
            dict_to_write['emoticon_list'] = row['emoticon list']
            dict_to_write['cashtag_list'] = row['cashtag list']
            dict_to_write['n-grams'] = row['n-grams']
            dict_to_write['engineered features'] = row['engineered features'][0]
            dict_as_string = json.dumps(dict_to_write) + "\n"
            f.write(dict_as_string)
    
def date_range_to_weekly_date_sets(start, end):
    """
    returns a list of sets, where each set
    contains the datetimes of the week that it corresponds to
    """
    date_sets = []
    current_date = start
    while current_date <= end:
        date_sets.append(set([current_date + timedelta(days=i) for i in range(7)]))
        current_date += timedelta(days=7)
    return date_sets

def split_train_test(df):
    """
    splits the total dataset into 80/20 split where 
    chronological order is maintained
    """

    # sort dataframe by dates low to high
    df = df.sort_values(by='date')
    df = df.reset_index(drop=True)

    # split the dataframe into 80/20 train and test
    train_df = df.iloc[:int(0.8*len(df))]
    test_df = df.iloc[int(0.8*len(df)):]

    return train_df, test_df

def split_test_by_week(df):
    """
    splits the test dataframe into a list of dataframes,
    where each dataframe contains the test data for a single week
    """

    start_date = df['date'].min()
    end_date = df['date'].max()

    date_sets = date_range_to_weekly_date_sets(start_date, end_date)

    test_dfs = []
    for date_set in date_sets:
        test_df = df[df['date'].isin(date_set)]
        test_dfs.append(test_df)

    return test_dfs

def test_df_writer(df_list, base_path):
    for test_df in df_list:
        print("wrote one test dataframe")
        test_path = test_path_compiler(test_df, base_path)
        write_df_to_engineered(test_df, test_path)

if __name__ == "__main__":
    ENGINEERED_PATH = '../data/pre_processed_data/engineered_total_train.txt'
    TEST_OUTPUT_FOLDER = '../data/train_test_data/weekly_split/test'
    TRAIN_OUTPUT_FOLDER = '../data/train_test_data/weekly_split/train'
    df, start_date, end_date = df_builder(ENGINEERED_PATH)
    train_df, test_df = split_train_test(df)
    test_dfs = split_test_by_week(test_df)
    
    # write train_df to file
    write_df_to_engineered(train_df, os.path.join(TRAIN_OUTPUT_FOLDER, 'train.dat'))

    # write all test_dfs to files
    test_df_writer(test_dfs, TEST_OUTPUT_FOLDER)