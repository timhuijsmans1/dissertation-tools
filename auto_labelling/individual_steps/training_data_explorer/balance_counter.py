from itertools import count


def balance_counter(path):
    count_dict = {'positive': 0, 'neutral': 0, 'negative': 0}
    with open(path, 'r') as f:
        for line in f:
            sentiment = line.split('\t')[-1].strip("\n ")
            count_dict[sentiment] += 1
    print(count_dict)

if __name__ == "__main__":
    SEMEVAL_PATH = './train/cleansed/twitter-train-cleansed-A.tsv'
    balance_counter(SEMEVAL_PATH)
