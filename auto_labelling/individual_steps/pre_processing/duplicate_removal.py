import json
import os
import regex
import re
import sys
import emoji as emoji_lib
import numpy as np
import line_profiler
import atexit
import datetime

from scipy import sparse

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

class duplicatePreProcessor:

    def __init__(
        self, 
        raw_data_path, 
        pre_processed_path, 
        output_path, 
        collection_data_path,
        cosine_threshold,
        high_freq_threshold,
        ):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.pre_processed_path = pre_processed_path
        self.collection_data_path = collection_data_path
        self.cosine_threshold = cosine_threshold
        self.high_freq_threshold = high_freq_threshold

        self.unique_tweet_vectors = []
        self.vocabulary = set()

    def emoji_extractor(self, tweet_text):
        emoji_list = []
        data = regex.findall(r'\X', tweet_text)
        for word in data:
            if any(char in emoji_lib.UNICODE_EMOJI['en'] for char in word):
                emoji_list.append(word)
        flags = regex.findall(u'[\U0001F1E6-\U0001F1FF]', tweet_text) 
        
        emoji_list = emoji_list + flags

        for emoji in emoji_list:
            tweet_text = tweet_text.replace(emoji, ' ') # replace with a space 
                                                        # to make sure that potential
                                                        # words around the emoji 
                                                        # separated
    
        return tweet_text, emoji_list

    def cashtag_extractor(self, tweet_text):
        cashtag_list = []
        cashtag_list = regex.findall(r'\$[a-zA-Z0-9]+', tweet_text)

        for cashtag in cashtag_list:
            tweet_text = tweet_text.replace(cashtag, '')

        return tweet_text, cashtag_list

    def string_processing(self, tweet_text):
        # replace all linebreaks and tabs in tweet by spaces
        tweet_text = tweet_text.replace('\n', ' ')
        # replace all tabs in tweet by spaces
        tweet_text = tweet_text.replace('\t', ' ')
        # remove RT tag from tweet
        tweet_text = tweet_text.replace('RT', '')
        # replace all multiple spaces in tweet by single spaces
        tweet_text = re.sub(r'\s+', ' ', tweet_text)
        # remove all web links from tweet
        tweet_text = re.sub(r'http\S+', '', tweet_text)
        # remove all usernames from tweet  
        tweet_text = re.sub(r'@\S+', '', tweet_text)
        # replace all underscores by a space
        tweet_text = tweet_text.replace('_', ' ')

        return tweet_text

    def tokenisation(self, tweet_text):

        # split tweet on spaces
        tweet_tokens = tweet_text.split(' ')

        # strip all punctuation of tokens
        tweet_tokens = [token.strip(',.!;:?-()# ') for token in tweet_tokens]

        # remove all tokens not containing any alphanumeric characters
        tweet_tokens = [token.lower() for token in 
                        tweet_tokens if re.search(r'[a-zA-Z0-9]', token)]

        # remove all empty tokens
        tweet_tokens = [token for token in tweet_tokens if token != '']

        # remove all numerical tokens
        non_numerical_tokens = []
        pattern_pos = r'[1-9]\d*(\.\d+)?'
        for token in tweet_tokens:
            if not re.search(pattern_pos, token):
                non_numerical_tokens.append(token)            

        return tweet_tokens, non_numerical_tokens

    def tweet_pre_processing(self, tweet_data):
        original_text = tweet_data['text']
        processed_text = self.string_processing(original_text)

        # extract and remove emojis and cashtags from the tweet text
        processed_text, emoticon_list = self.emoji_extractor(processed_text)
        processed_text, cashtag_list = self.cashtag_extractor(processed_text)

        tweet_tokens, non_numerical_tokens = self.tokenisation(processed_text)
        tweet_data = {
                'original_text': original_text,
                'created_at': tweet_data['created_at'],
                'processed_tokens': tweet_tokens,
                'non_numerical_tokens': non_numerical_tokens,
                'emoticon_list': emoticon_list,
                'cashtag_list': cashtag_list
        }
        return tweet_data

    def pre_processing(self):
        with open(self.raw_data_path, 'r') as f_in, open(self.pre_processed_path, 'w') as f_out:
            
            self.tweet_count = 0
            while True:
                line = f_in.readline()
                if not line:
                    break
                else:
                    tweet_data = json.loads(line)
                    original_text = tweet_data['text']
                    
                    # exclude Tweets with 8+ repeated cashtags
                    pattern = re.compile(r"\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+\s\$\w+")
                    if pattern.search(original_text): 
                        continue

                    # run pre_processing on individual tweets
                    tweet_data_to_write = self.tweet_pre_processing(tweet_data)
                    non_numerical_tokens = tweet_data_to_write['non_numerical_tokens']
                    self.vocabulary |= set(non_numerical_tokens)
                    
                    # write tweet data to the pre_processed_data file
                    f_out.write(json.dumps(tweet_data_to_write) + '\n')

                    self.tweet_count += 1
                    
                    # print tweet count every 1000 tweets
                    if self.tweet_count % 100 == 0:
                        print('Processed {} tweets'.format(self.tweet_count))

        # write collection data to disk
        collection_data = {'tweet_count': self.tweet_count,
                            'vocabulary': list(self.vocabulary)}
        with open(self.collection_data_path, 'w+') as f:
            json.dump(collection_data, f)
        return

    def load_collection_data(self):
        with open(self.collection_data_path, 'r') as f:
            collection_data = json.load(f)
            self.tweet_count = collection_data['tweet_count']
            self.vocabulary = collection_data['vocabulary']
        return

    def get_word2index(self):
        self.word2index = {}
        for i, word in enumerate(self.vocabulary):
            self.word2index[word] = i

        # swap all keys and values of word2index
        self.index2word = {v: k for k, v in self.word2index.items()}
        
        return

    def dense_vector_from_tokens(self, tokens):
        dense_tweet_vector = np.zeros(len(self.vocabulary))
        for token in tokens:
            token_index = self.word2index[token]
            dense_tweet_vector[token_index] += 1
        return dense_tweet_vector

    def cosine_similarity_check(self, threshold, vector_1, vector_2):
        """
        calculate cosine similarity between vector 1 and 2: return
        True if higher/equal than threshold and False if lower than threshold
        """
        cosine_similarity = np.dot(vector_1, vector_2) \
            / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))

        return cosine_similarity >= threshold
    
    def duplicate_check(self, sparse_vector_list, vector_to_check):
        # assume that the tweet is unique until tested
        new_tweet = True
        for i, vector in enumerate(sparse_vector_list):
            dense_existing_vector = vector[0].todense()[0]
            if self.cosine_similarity_check(
                        self.cosine_threshold, 
                        dense_existing_vector,
                        vector_to_check
                ):
                new_tweet = False
                break

        return new_tweet, i

    def sparse_in_list(self, list, sparse_vector):
        for vector in list:
            if (vector.todense() == sparse_vector.todense()).all():
                return True
        return False
                            
    def duplicate_filter(self):
        sparse_vectors = []
        high_freq_vectors = []
        with open(self.pre_processed_path, 'r') as f_in:

            with open(self.output_path, 'w') as f_out:
                count = 0
                while True:
                    # limit the size of the high freq vectors to avoid 
                    # time issues
                    if len(high_freq_vectors) > 200:
                        high_freq_vectors = []
                    line = f_in.readline()
                    if not line:
                        break
                    else:
                        tweet_data = json.loads(line)
                        pre_processed_text = (
                            tweet_data['non_numerical_tokens']
                        )
                        dense_new_vector = self.dense_vector_from_tokens(
                                                            pre_processed_text
                        )

                        # check if duplicate or similar, always add the first
                        # vector to the list of sparse vectors
                        if sparse_vectors:
                            new_tweet, duplicate_index = self.duplicate_check(
                                                sparse_vectors, 
                                                dense_new_vector,
                            )
                            # if the tweet vector is not similar, add it to the list
                            # and write the corresponding tweet data to the output file
                            if new_tweet:
                                sparse_vectors.append([sparse.coo_array(dense_new_vector), 1])
                                f_out.write(line)
                            # if the tweet vector is similar, increase the frequency of the
                            # vector in the list and 
                            else:
                                sparse_vectors[duplicate_index][1] += 1
                                if sparse_vectors[duplicate_index][1] > self.high_freq_threshold:
                                    if not self.sparse_in_list(high_freq_vectors, sparse_vectors[duplicate_index][0]):
                                        high_freq_vectors.append(sparse_vectors[duplicate_index][0])
                        else:
                            sparse_vectors.append([sparse.coo_array(dense_new_vector), 1])
                            f_out.write(line)
                    
                        count += 1

                        if count % 1000 == 0:
                            # this stores the history of high freq tweets in the 
                            # sparse vector list to use in the next 1000 duplicate 
                            # checks. the freq of one does not matter, as these tweets 
                            # will remain in the high freq tweets anyways.
                            sparse_vectors = [[vector, 1] for vector in high_freq_vectors]
                            print("tweets in most recent tweets : ", len(sparse_vectors))
                        if count % 100 == 0:    
                            print("tweets processed: ", count)
                            print("tweets in high freq tweets: ", len(high_freq_vectors))
                            print("tweets in most recent tweets : ", len(sparse_vectors))
        os.remove(self.pre_processed_path) 
        return                        

                        
if __name__ == "__main__":
    FULL_DATA_PATH = "../../collector_data/search_results.txt"
    PREPROCESSED_PATH = "../data/preprocessed_data/preprocessed.txt"
    COLLECTION_DATA_PATH = "../data/preprocessed_data/collection_data.txt"
    OUTPUT_PATH = "../data/preprocessed_data/final_data.txt"
    COSINE_SIM_THRESHOLD = 0.6
    HIGH_FREQ_THRESHOLD = 10
    
    pre_processor = duplicatePreProcessor(
                        FULL_DATA_PATH, 
                        PREPROCESSED_PATH, 
                        OUTPUT_PATH, 
                        COLLECTION_DATA_PATH,
                        COSINE_SIM_THRESHOLD,
                        HIGH_FREQ_THRESHOLD
    )
    pre_processor.pre_processing()
    pre_processor.get_word2index()
    pre_processor.duplicate_filter()