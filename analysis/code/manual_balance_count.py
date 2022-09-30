import json
import os

def tweet_printer(daily_path_list):
    for i, path in enumerate(daily_path_list):
        print(i, 'is een kankerbitch')
        with open(path, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                print(tweet['original_text'])
                print('-'* 20)
            print('-' * 50)
            print('-' * 50)
            print('-' * 50)
            print('-' * 50)

if __name__ == "__main__":
    DATA_FOLDER = "../../labelled_data2predictions/data/pipeline_output/ABNB/engineered_data"
    path_list = [os.path.join(DATA_FOLDER, filename) for filename in os.listdir(DATA_FOLDER) if filename.endswith(".txt")]
    tweet_printer(path_list)