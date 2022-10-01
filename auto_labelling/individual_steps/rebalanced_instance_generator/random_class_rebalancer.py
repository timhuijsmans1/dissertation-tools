import json
import random

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
    