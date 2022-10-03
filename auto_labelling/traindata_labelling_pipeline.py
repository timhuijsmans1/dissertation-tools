import datetime
import json
import os

from twarc import Twarc2, expansions

from individual_steps.data_collector.collector import TweetCollector
from individual_steps.pre_processing.duplicate_removal import duplicatePreProcessor
from individual_steps.data_labelling.data_labeller import dataLabeller
from individual_steps.rebalanced_instance_generator.random_class_rebalancer import *
from individual_steps.rebalanced_instance_generator.tweet2instance import *

def prompt_utc_time():
    print("Provide start date")
    start_year = input("Year (yyyy): ")
    start_month = input("Month (mm): ")
    start_day = input("Day (dd): ")
    print("Provide end date")
    end_year = input("Year (yyyy): ")
    end_month = input("Month (mm): ")
    end_day = input("Day (dd): ")

    try:
        start_datetime = datetime.datetime(
                                int(start_year), 
                                int(start_month),
                                int(start_day), 0, 0, 0, 0, 
                                datetime.timezone.utc
        )
        end_datetime = datetime.datetime(
                                int(end_year), 
                                int(end_month),
                                int(end_day), 0, 0, 0, 0, 
                                datetime.timezone.utc
        )
    except:
        raise Exception(
            "Invalid date arguments, please re-run and try again"
        )
    
    if start_datetime > end_datetime:
        raise Exception(
            "End date can not be before start date," 
            + " please re-run and try again"
        )

    return start_datetime, end_datetime

def valid_labelling_method(labelling_method):
    labelling_methods = {'union', 'emoticon', 'lexicon'}
    if labelling_method not in labelling_methods:
        return False
    else:
        return True

def rebalanced_instance_generator(
    labelled_data_path, 
    rebalanced_data_path, 
    engineered_data_path, 
    vocabulary_path, 
    instances_path
    ):
    # run rebalancing
    lowest_count = class_balance_counter(labelled_data_path)
    random_rebalancer(labelled_data_path, rebalanced_data_path, lowest_count)
    # run instance generation
    feature_engineering = featureEngineering(rebalanced_data_path, engineered_data_path)
    total_vocabulary = feature_engineering.feature_updating()
    write_vocabulary(vocabulary_path, total_vocabulary)
    sparse_total_matrix, total_labels = build_sparse_matrix(engineered_data_path, total_vocabulary)
    write_data_set(instances_path, sparse_total_matrix, total_labels)

def main():
    tweet_collector = TweetCollector(
                        NASDAQ_100_PATH,
                        FULL_NASDAQ_PATH,
                        NYSE_PATH,
                        EMOTICON_LIST,
                        CLIENT,
                        START_TIME,
                        END_TIME,
                        SEARCH_RESULT_PATH
    )
    tweet_collector.execute_combined_search()

    # pre-process and duplicate removal 
    pre_processor = duplicatePreProcessor(
                        SEARCH_RESULT_PATH, 
                        PREPROCESSED_PATH, 
                        NON_DUPLICATE_PATH, 
                        COLLECTION_DATA_PATH,
                        COSINE_SIM_THRESHOLD,
                        HIGH_FREQ_THRESHOLD
    )
    pre_processor.pre_processing()
    pre_processor.get_word2index()
    pre_processor.duplicate_filter()

    # label Tweets with union method
    data_labeller = dataLabeller(
        FINANCIAL_LEXICON_PATH, 
        LABELLED_DATA_PATH, 
        NEG_INDICATOR_PATH,
        POS_EMOTICON_LIST, 
        NEG_EMOTICON_LIST
    )
    data_labeller.file_labeller(NON_DUPLICATE_PATH, method='union')

    # label Tweets with emoticon method
    data_labeller = dataLabeller(
        FINANCIAL_LEXICON_PATH, 
        LABELLED_EMO_DATA_PATH, 
        NEG_INDICATOR_PATH,
        POS_EMOTICON_LIST, 
        NEG_EMOTICON_LIST
    )
    data_labeller.file_labeller(NON_DUPLICATE_PATH, method='emoticon')

    # run union instance generation
    rebalanced_instance_generator(
        LABELLED_DATA_PATH,
        REBALANCED_DATA_PATH,
        ENGINEERED_DATA_PATH,
        VOCABULARY_PATH,
        INSTANCES_PATH
    )
    # run emoticon instance generation
    rebalanced_instance_generator(
        LABELLED_EMO_DATA_PATH,
        REBALANCED_EMO_DATA_PATH,
        ENGINEERED_EMO_DATA_PATH,
        VOCABULARY_EMO_PATH,
        INSTANCES_EMO_PATH
    )

if __name__ == "__main__":
    # Tweet collection paths
    INDIVIDUAL_PATH = "individual_steps/"
    NASDAQ_100_PATH = INDIVIDUAL_PATH + "data_collector/nasdaq_100_listings.csv"
    FULL_NASDAQ_PATH = INDIVIDUAL_PATH + "data_collector/nasdaq-listed-symbols.json"
    NYSE_PATH = INDIVIDUAL_PATH + "data_collector/nyse-listed.json"
    SEARCH_RESULT_PATH = "collector_data/search_results.txt"
    BEARER = os.environ.get("BEARER")
    CLIENT = Twarc2(bearer_token=BEARER)
    POS_EMOTICON_LIST = ["😀", "😃", "😄", "😁", "🙂"]
    NEG_EMOTICON_LIST = [
        "😡", "😤", "😟", "😰", "😨", "😖",
        "😩", "🤬", "😠", "💀", "👎", "📉"
    ]
    EMOTICON_LIST = POS_EMOTICON_LIST + NEG_EMOTICON_LIST
    START_TIME, END_TIME = prompt_utc_time()

    # duplicate removal paths
    PREPROCESSED_PATH = "collector_data/preprocessed.txt"
    COLLECTION_DATA_PATH = "collector_data/collection_metadata.txt"
    NON_DUPLICATE_PATH = "collector_data/non_duplicate_data.txt"
    COSINE_SIM_THRESHOLD = 0.6
    HIGH_FREQ_THRESHOLD = 10

    # labelling paths
    LABELLED_DATA_PATH = "collector_data/labelled_data.txt"
    LABELLED_EMO_DATA_PATH = "collector_data/labelled_emo_data.txt"
    FINANCIAL_LEXICON_PATH = "individual_steps/data/fin_sent_lexicon/lexicons/lexiconWNPMINW.csv"
    NEG_INDICATOR_PATH = "individual_steps/data_labelling/negation_ind.txt"

    # instance generation paths
    REBALANCED_DATA_PATH = "collector_data/rebalanced_labelled_data.txt"
    ENGINEERED_DATA_PATH = "collector_data/rebalanced_engineered_labelled_data.txt"
    VOCABULARY_PATH = "collector_data/prediction_input_files/vocabulary.txt"
    INSTANCES_PATH = "collector_data/prediction_input_files/train_instances.txt"

    REBALANCED_EMO_DATA_PATH = "collector_data/rebalanced_labelled_emo_data.txt"
    ENGINEERED_EMO_DATA_PATH = "collector_data/rebalanced_engineered_labelled_emo_data.txt"
    VOCABULARY_EMO_PATH = "collector_data/prediction_input_files/emo_vocabulary.txt"
    INSTANCES_EMO_PATH = "collector_data/prediction_input_files/emo_train_instances.txt"

    main()