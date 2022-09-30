import json
import os
import datetime
import quapy as qp

from collection_labelling_pipeline import build_sparse_matrix, prevalence_calculator, \
    prediction_writer, load_vocabulary, utc_to_eastern
from quapy.method.aggregative import SVMNKLD, SVMQ, SVMRAE
from pytz import utc
from pytz import timezone
eastern = timezone('US/Eastern')

def get_list_of_dates(single_ticker_path):
    list_of_date_files = [date for date in os.listdir(single_ticker_path) if date[0] != "."]
    list_of_dates = [date.split(".")[0] for date in list_of_date_files]
    list_of_dates = sorted(list_of_dates)
    return list_of_dates

def get_file_paths_per_day(data_folder):
    tickers = [ticker for ticker in os.listdir(data_folder) if ticker[0] != "."]
    single_ticker_path = os.path.join(data_folder, tickers[0], 'engineered_data')
    list_of_dates = get_list_of_dates(single_ticker_path)

    all_paths_per_day = {}
    for date in list_of_dates:
        daily_ticker_paths = []
        for ticker in tickers:
            daily_ticker_paths.append(os.path.join(data_folder, ticker, 'engineered_data', date + '.txt'))
        all_paths_per_day[date] = daily_ticker_paths

    return all_paths_per_day

def split_engineered_pre_intra(daily_engineered_paths, market_opening):
    """
    single_day_engineered_paths: list of paths to engineered data of each ticker
                                 for a single day.
    market_opening: datetime object of the market opening time.
    """
    pre_market_lines = []
    intra_market_lines = []
    for path in daily_engineered_paths:
        with open(path, 'r') as f_in:
            for line in f_in:
                tweet_data = json.loads(line)
                # we select only those keys that we need, to save memory
                tweet_data_to_return = {
                    'n-grams': tweet_data['n-grams'],
                    'engineered features': tweet_data['engineered features'],
                }
                created_at_utc = tweet_data['created_at']
                created_at_eastern = utc_to_eastern(created_at_utc)
                if created_at_eastern < market_opening:
                    pre_market_lines.append(tweet_data_to_return)
                else:
                    intra_market_lines.append(tweet_data_to_return)
    return {'pre': pre_market_lines, 'intra': intra_market_lines}

def train_model(
    union_data_path, 
    emoticon_data_path, 
    mock_test_path
    ):

    union_data = qp.data.Dataset.load(union_data_path, mock_test_path, qp.data.reader.from_sparse)
    emoticon_data = qp.data.Dataset.load(emoticon_data_path, mock_test_path, qp.data.reader.from_sparse)
    union_model = SVMRAE()
    union_model.fit(union_data.training)
    emoticon_model = SVMRAE()
    emoticon_model.fit(emoticon_data.training)

    return union_model, emoticon_model

def daily_aggregate_sentiment(
    all_day_paths, 
    output_path, 
    union_voc_path, 
    emoticon_voc_path, 
    union_model, 
    emoticon_model
    ):
    for i, date_string in enumerate(all_day_paths.keys()):
        print(f"day {i} out of {len(all_day_paths.keys())}")
        daily_engineered_paths = all_day_paths[date_string]
        datetime_object = datetime.datetime.strptime(date_string, '%Y-%m-%d')
        datetime_object = datetime_object.replace(tzinfo=eastern)
        daily_folder_name = date_string
        market_opening = datetime_object.replace(hour=9, minute=30)

        # this takes all ticker paths of a single day, and splits
        # the tweets into pre and intra market based on their timestap
        pre_and_intra_tweet_dict = split_engineered_pre_intra(daily_engineered_paths, market_opening)

        prediction_day_folder = os.path.join(output_path, daily_folder_name)
        os.mkdir(prediction_day_folder)
        vocabulary_union = load_vocabulary(union_voc_path) 
        vocabulary_emoticon = load_vocabulary(emoticon_voc_path)

        for moment_of_day in pre_and_intra_tweet_dict.keys():
            print(moment_of_day, ' market')
            print('building sparse matrices')
            sparse_total_matrix_union = build_sparse_matrix(pre_and_intra_tweet_dict[moment_of_day], vocabulary_union).T
            sparse_total_matrix_emoticon = build_sparse_matrix(pre_and_intra_tweet_dict[moment_of_day], vocabulary_emoticon).T
            print('predicting labels')
            prevalence_union, labels_union = prevalence_calculator(union_model, sparse_total_matrix_union)
            prevalence_emoticon, labels_emoticon = prevalence_calculator(emoticon_model, sparse_total_matrix_emoticon)
            
            # to int in order to write to json
            labels_union = [int(label) for label in labels_union]
            labels_emoticon = [int(label) for label in labels_emoticon]
            predictions = {
                'union': {'prevalence': prevalence_union, 'labels': labels_union},
                'emoticon': {'prevalence': prevalence_emoticon, 'labels': labels_emoticon}
            }
            print(predictions)
            prediction_day_path = os.path.join(prediction_day_folder, moment_of_day + ".json")
            prediction_writer(predictions, prediction_day_path)
        print('-' * 60)


if __name__ == "__main__":
    qp.environ['SVMPERF_HOME'] = '../svm_perf_quantification'
    
    # global variables taken from the pipeline input, 
    # as these are used for the model inputs (vocabs and data)
    INPUT_FOLDER = '../../data/pipeline_input'
    UNION_VOCABULARY_PATH = os.path.join(INPUT_FOLDER, 'vocabularies/vocabulary_union_cashtag.txt')
    EMOTICON_VOCABULARY_PATH = os.path.join(INPUT_FOLDER, 'vocabularies/vocabulary_emoticon_cashtag.txt')
    TRAIN_DATA_UNION_PATH = os.path.join(INPUT_FOLDER, 'train_data/train_union_cashtag.dat')
    TRAIN_DATA_EMOTICON_PATH = os.path.join(INPUT_FOLDER, 'train_data/train_emoticon_cashtag.dat')
    MOCK_TEST_FILE_PATH = os.path.join(INPUT_FOLDER, 'train_data/mock_test_file.dat')

    # global variables specific for this script
    DATA_FOLDER = "../../data/pipeline_output" # this is the folder we use for the input data
    OUTPUT_PATH = "../../data/aggregate_sentiment_output/^NDX/predictions"
    all_paths_per_day = get_file_paths_per_day(DATA_FOLDER)
    union_model, emoticon_model = train_model(
                                    TRAIN_DATA_UNION_PATH,
                                    TRAIN_DATA_EMOTICON_PATH,
                                    MOCK_TEST_FILE_PATH
    )
    daily_aggregate_sentiment(
        all_paths_per_day, 
        OUTPUT_PATH, 
        UNION_VOCABULARY_PATH, 
        EMOTICON_VOCABULARY_PATH, 
        union_model, 
        emoticon_model
    )
