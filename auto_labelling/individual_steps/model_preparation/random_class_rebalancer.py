from cProfile import label
import json
import random
import os

from pyrsistent import l

def class_rebalancer(lowest_count, input_path, output_path):
    neg_count = 0
    pos_count = 0
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            label = int(line.split(" ")[0])
            if label == 1:
                if pos_count < lowest_count:
                    f_out.write(line)
                    pos_count += 1
            if label == -1:
                if neg_count < lowest_count:
                    f_out.write(line)
                    neg_count += 1

def random_rebalancer(input_path, output_path, amount_to_write):
    neg_tweets = []
    pos_tweets = []
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            label = int(json.loads(line)['label'])
            if label == 1:
                pos_tweets.append(line)
            if label == -1:
                neg_tweets.append(line)
        
        pos_to_write = random.sample(pos_tweets, amount_to_write)
        neg_to_write = random.sample(neg_tweets, amount_to_write)
        tweets_to_write = pos_to_write + neg_to_write
        print('writing a total of {} tweets'.format(len(tweets_to_write)))
        for tweet in tweets_to_write:
            f_out.write(tweet)
            
def class_balance_counter(path):
    label_counts = {'positive': 0, 'negative': 0}
    with open(path, 'r') as f:
        for line in f:
            label = json.loads(line)['label']
            if label == 1:
                label_counts['positive'] +=  1
            if label == -1:
                label_counts['negative'] += 1
    
    lowest_count = min(label_counts.values())
    return lowest_count

if __name__ == "__main__":
    LABELLED_DATA = "../data/pre_processed_data/unbalanced/labelled_data_lexicon_cashtag.txt"
    OUTPUT_PATH = "../data/pre_processed_data/rebalanced/labelled_data_lexicon_cashtag_rebalanced.txt" 
    # lowest_count = 28599 # the amount of negative tweets in the auto labelled set
    lowest_count = class_balance_counter(LABELLED_DATA)
    lowest_count = 14291 # the amount of negative tweets in the V2 auto labelled set
    random_rebalancer(LABELLED_DATA, OUTPUT_PATH, lowest_count)
    