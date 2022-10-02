import json
import os
import re

class dataLabeller:

    def __init__(
            self, 
            lexicon_path, 
            output_path,
            negation_file_path,
            pos_emoticon, 
            neg_emoticon
        ):
        if os.path.exists(output_path):
            os.remove(output_path) # removes the existing labelled file to allow
                                   # for append mode in writing.

        self.lexicon = self.lexicon_reader(lexicon_path)
        self.output_path = output_path
        self.pos_emoticon = pos_emoticon
        self.neg_emoticon = neg_emoticon
        self.negation_indicators = self.negation_indicator_loader(negation_file_path)
        self.neutral_emo_count = 0
        self.removed_crypto_tweet = 0
    
    def negation_indicator_loader(self, negation_file_path):
        negation_indicators = set()
        with open(negation_file_path, 'r') as f:
            for line in f:
                negation_indicators.add(line.strip('\n '))
        return negation_indicators

    def lexicon_reader(self, lexicon_path):
        lexicon = {}
        with open(lexicon_path, 'r') as f:
            header = f.readline()

            while True:
                line = f.readline()
                if not line:
                    break
                else:
                    word, score = line.split(",")[1:]
                    word = word.strip("\"")
                    score = float(score.strip("\n"))
                    lexicon[word] = score
        
        return lexicon

    def lexicon_labeller(self, tweet_text):
        # calculate the total tweet sentiment score
        tweet_score = 0
        for token in tweet_text:
            token_score = self.lexicon.get(token, 0)
            tweet_score += token_score

        # set sentiment labels based of score
        if tweet_score > 0:
            label = 1
        elif tweet_score < 0:
            label = -1
        else:
            label = 0
        
        return label, tweet_score
    
    def emoticon_labeller(self, emoticon_list):
        contains_pos = False
        contains_neg = False
        emoticon_score = 0
        used_emoticons = [] # keeps track of the emoticons used for score calculation
        for emoticon in emoticon_list:
            if emoticon in self.pos_emoticon:
                used_emoticons.append(emoticon)
                emoticon_score += 1
                contains_pos = True
            if emoticon in self.neg_emoticon:
                used_emoticons.append(emoticon)
                emoticon_score -= 1
                contains_neg = True
        if emoticon_score > 0:
            label = 1
        elif emoticon_score < 0:
            label = -1
        elif emoticon_score == 0 and used_emoticons:
            label = 0
            self.neutral_emo_count += 1
        else:
            label = None
        mixed_emotions = contains_pos * contains_neg
        return label, emoticon_score, used_emoticons, mixed_emotions

    def negator(self, token):
        return "neg_" + token

    def negation_handling(self, tokens):
        negated_tokens = []
        
        i = 0
        while i < len(tokens):
            if tokens[i] in self.negation_indicators:
                
                # antepenultimate token is negation indicator
                if i + 3 == len(tokens):
                    negated_tokens.append(self.negator(tokens[i + 1]))
                    negated_tokens.append(self.negator(tokens[i + 2]))
                    break

                # penultimate token is negation indicator
                elif i + 2 == len(tokens):
                    negated_tokens.append(self.negator(tokens[i + 1]))
                    break

                # last token is negation indicator
                elif i + 1 == len(tokens):
                    break
                    
                else:
                    negated_tokens.append(self.negator(tokens[i + 1]))
                    negated_tokens.append(self.negator(tokens[i + 2]))
                    i += 3 # skip next two tokens

            else:
                negated_tokens.append(tokens[i])
                i += 1

        return negated_tokens

    def num_handling(self, tokens):
        pattern_pos = r'^\+[1-9]\d*(\.\d+)?'
        pattern_neg = r'^\-[1-9]\d*(\.\d+)?'
        i = 0
        while i < len(tokens):
            if re.search(pattern_pos, tokens[i]):
                tokens[i] = 'posnum'
            elif re.search(pattern_neg, tokens[i]):
                tokens[i] = 'negnum'
            i += 1
        return tokens

    def neg_and_num_execution(self, tweet_data):
        tweet_tokens = tweet_data['processed_tokens']
        tweet_tokens = self.num_handling(tweet_tokens)
        tweet_tokens = self.negation_handling(tweet_tokens)
        tweet_data['processed_tokens'] = tweet_tokens

        return tweet_data

    def label_writer(
        self, 
        output_path, 
        dict_to_write
        ):
        string_of_dict = json.dumps(dict_to_write) + "\n"
        with open(output_path, 'a+') as f:
            f.write(string_of_dict)

        return

    def emoticon_lexicon_executor(self, dict_to_write):
        tweet_tokens = dict_to_write['processed_tokens']
        emoticon_list = dict_to_write['emoticon_list']
        # label the tweet
        lexicon_label, lexicon_score = self.lexicon_labeller(tweet_tokens)
        emoticon_label, emoticon_score, used_emoticons, mixed_emotions = self.emoticon_labeller(emoticon_list)

        # skip tweets with a neutral emoticon sum
        if lexicon_label == emoticon_label and emoticon_label != 0:
            dict_to_write['label'] = lexicon_label
        
        return dict_to_write
    
    def lexicon_executor(self, dict_to_write):
        tweet_tokens = dict_to_write['processed_tokens']
        # label the tweet
        lexicon_label, lexicon_score = self.lexicon_labeller(tweet_tokens)
        if lexicon_label != 0:    
            dict_to_write['label'] = lexicon_label
        
        return dict_to_write

    def emoticon_executor(self, dict_to_write):
        emoticon_list = dict_to_write['emoticon_list']
        # label the tweet
        emoticon_label, emoticon_score, used_emoticons, mixed_emotions = self.emoticon_labeller(emoticon_list)
        if emoticon_label != None and emoticon_label != 0: # skip neutral and none tweets
            dict_to_write['label'] = emoticon_label
        
        return dict_to_write

    def contains_spamwords(self, tweet_text_string):
        tweet_text_string = tweet_text_string.lower()
        spam_words = [
            'crypto', 
            'discord', 
            'altcoin', 
            '$eth', 
            '$shib',
            '$doge',
            'alt', 
            'coinbase'
        ]
        for word in spam_words:
            if word in tweet_text_string:
                self.removed_crypto_tweet += 1
                return True
        return False

    def file_labeller(self, path, method='union'):
        class_balance_counter = {-1: 0, 0: 0, 1: 0}

        with open(path, 'r') as f:
            count = 1
            for line in f:

                dict_to_write = json.loads(line)
                dict_to_write = (
                    self.neg_and_num_execution(dict_to_write)
                )
                original_text_string = dict_to_write['original_text']
                
                # skip the tweet if it contains any of the spam words
                if self.contains_spamwords(original_text_string):
                    continue

                if method == 'union':
                    dict_to_write = self.emoticon_lexicon_executor(dict_to_write)
                elif method == 'lexicon':
                    dict_to_write = self.lexicon_executor(dict_to_write)
                elif method == 'emoticon':
                    dict_to_write = self.emoticon_executor(dict_to_write)
                
                if 'label' in dict_to_write:
                    label = dict_to_write['label']
                    self.label_writer(self.output_path, dict_to_write)

                    if label == 1:
                        class_balance_counter[1] += 1
                    elif label == -1:
                        class_balance_counter[-1] += 1
                    elif label == 0:
                        class_balance_counter[0] += 1

                count += 1
                if count % 1000 == 0:
                    print(count)
        print(class_balance_counter)

        return

if __name__ == "__main__":
    # global variables
    DATA_PATH = (
        "../data/preprocessed_data/non_duplicate_data.txt"
    )
    OUTPUT_PATH = "../data/labelled_data/labelled_data_union.txt"
    FINANCIAL_LEXICON_PATH = "../data/fin_sent_lexicon/lexicons/lexiconWNPMINW.csv"
    NEG_INDICATOR_PATH = "./negation_ind.txt"
    POS_EMOTICON_LIST = ["😀", "😃", "😄", "😁", "🙂"]
    NEG_EMOTICON_LIST = ["😡", "😤", "😟", "😰", "😨", "😖", "😩", "🤬", "😠", "💀", "👎", "📉"]

    # exucute data labelling
    data_labeller = dataLabeller(
        FINANCIAL_LEXICON_PATH, 
        OUTPUT_PATH, 
        NEG_INDICATOR_PATH,
        POS_EMOTICON_LIST, 
        NEG_EMOTICON_LIST
    )
    data_labeller.file_labeller(DATA_PATH, method='union')
