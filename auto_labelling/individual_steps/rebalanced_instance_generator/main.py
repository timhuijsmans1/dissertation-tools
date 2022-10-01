from random_class_rebalancer import *
from tweet2instance import *

def run_rebalancing(
            union_labelled, 
            emoticon_labelled, 
            union_rebalanced, 
            emoticon_rebalanced
    ):
    lowest_count = class_balance_counter(union_labelled)
    random_rebalancer(union_labelled, union_rebalanced, lowest_count)
    random_rebalancer(emoticon_labelled, emoticon_rebalanced, lowest_count)

def run_instance_generation(
            rebalanced_path,
            engineered_path, 
            vocabulary_path, 
            instance_out_path
    ):
    feature_engineering = featureEngineering(rebalanced_path, engineered_path)
    total_vocabulary = feature_engineering.feature_updating()
    write_vocabulary(vocabulary_path, total_vocabulary)
    sparse_total_matrix, total_labels = build_sparse_matrix(engineered_path, total_vocabulary)
    write_data_set(instance_out_path, sparse_total_matrix, total_labels)

if __name__ == "__main__":
    """INPUT FOLDERS"""
    # raw data
    FULL_LABELLED_DATA_UNION = '../../data/labelled_data/labelled_data_union_cashtag.txt'
    FULL_LABELLED_DATA_EMOTICON = '../../data/labelled_data/labelled_data_emoticon_cashtag.txt'
    
    # rebalanced
    REBALANCED_DATA_UNION = '../../data/rebalanced_labelled_data/rebalanced_union_cashtag.txt'
    REBALANCED_DATA_EMOTICON = '../../data/rebalanced_labelled_data/rebalanced_emoticon_cashtag.txt'
    
    # instance generation and training data
    ENGINEERED_UNION = '../../data/pipeline_input/engineered_data/engineered_union_cashtag.txt'
    VOCABULARY_UNION = '../../data/pipeline_input/vocabularies/vocabulary_union_cashtag.txt'
    INSTANCES_UNION = '../../data/pipeline_input/train_data/train_union_cashtag.dat'
    ENGINEERED_EMOTICON = '../../data/pipeline_input/engineered_data/engineered_emoticon_cashtag.txt'
    VOCABULARY_EMOTICON = '../../data/pipeline_input/vocabularies/vocabulary_emoticon_cashtag.txt'
    INSTANCES_EMOTICON = '../../data/pipeline_input/train_data/train_emoticon_cashtag.dat'

    run_rebalancing(
        FULL_LABELLED_DATA_UNION,
        FULL_LABELLED_DATA_EMOTICON,
        REBALANCED_DATA_UNION,
        REBALANCED_DATA_EMOTICON
    )

    run_instance_generation(
        REBALANCED_DATA_UNION,
        ENGINEERED_UNION,
        VOCABULARY_UNION,
        INSTANCES_UNION
    )

    run_instance_generation(
        REBALANCED_DATA_EMOTICON,
        ENGINEERED_EMOTICON,
        VOCABULARY_EMOTICON,
        INSTANCES_EMOTICON
    )


    