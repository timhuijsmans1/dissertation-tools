import json
import re

def filter_for_repeated_cashtags(path_in, path_out):
    count = 0
    with open(path_in, 'r') as f_in, open(path_out, 'w') as f_out:
        pattern = re.compile(r"\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+")
        for line in f_in:
            tweet = json.loads(line)
            # only keep tweets with less than 10 repeated cashtags
            if not pattern.search(tweet['text']):
                f_out.write(line)
            else:
                count += 1
    print(f"removed {count} tweets")

if __name__ == '__main__':
    DUPLICATE_REMOVED_DATA = "../data/preprocessed_data/final_data_july25.txt"
    CASHTAG_REPETITION_REMOVED = "../data/preprocessed_data/final_data_july25_cashtag_repetition.txt"
    filter_for_repeated_cashtags(DUPLICATE_REMOVED_DATA, CASHTAG_REPETITION_REMOVED)