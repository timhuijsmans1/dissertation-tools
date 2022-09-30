import json
import numpy as np
import re

from scipy import sparse
from nltk import bigrams

class features2Instance:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.word2index = self.get_word2index()

    def get_word2index(self):
        word2index = {}
        for i, word in enumerate(self.vocabulary):
            word2index[word] = i
        
        return word2index 

    def get_dense_vec(self, tweet_tokens, feature_dictionary):
        dense_tweet_vector = np.zeros(len(self.vocabulary))
        engineered_feature_values = np.fromiter(feature_dictionary.values(), dtype=float)
        
        for token in tweet_tokens:
            # OOV tokens will not be considered in the test data
            token_index = self.word2index.get(token, None)
            if token_index != None:
                dense_tweet_vector[token_index] += 1
        
        return np.concatenate((dense_tweet_vector, engineered_feature_values))

    def get_sparse_matrix(self, dense_vec):
        return sparse.coo_matrix(dense_vec)

    def get_sparse_matrix_from_tweet(self, tweet_tokens, feature_dictionary):
        dense_vec = self.get_dense_vec(tweet_tokens, feature_dictionary)
        return self.get_sparse_matrix(dense_vec)

class featureEngineering:

    def __init__(self, tweet_path, engineered_output_path):
        self.input_path = tweet_path
        self.output_path = engineered_output_path
        self.vocabulary = {}

    def post_num_and_negation_strip(self, tokens):
        return [token.strip(',.!;:?()#&- ') for token in tokens]

    def count_all_caps(self, original_text):
        pattern = re.compile(r"\s[A-Z]+\s")
        all_caps_tokens = re.findall(pattern, original_text)
        long_caps_tokens = [token for token in all_caps_tokens if len(token.strip("\t\n-+&$% ")) > 1]
        count = len(long_caps_tokens)
        return count

    def count_elongated(self, tokens):
        pattern = re.compile(r"(.)\1{2}")
        return len([token for token in tokens if pattern.search(token)])

    def count_negated(self, tokens):
        return len([token for token in tokens if "neg_" in token])

    def count_emoticons(self, emoticon_list):
        return len(emoticon_list)

    def find_bi_grams(self, tokens):
        bigrm = [" ".join(bigram_tokens) for bigram_tokens in bigrams(tokens)]
        tokens += bigrm
        return tokens

    def feature_generator(self, tweet_data):
        feature_dict = {}
        tokens = tweet_data['processed_tokens']
        original_text = tweet_data['original_text']
        emoticon_list = tweet_data['emoticon_list']

        stripped_tokens = self.post_num_and_negation_strip(tokens)
        feature_dict['all_caps_count'] = self.count_all_caps(original_text)
        feature_dict['elongated_count'] = self.count_elongated(stripped_tokens)
        feature_dict['negated_count'] = self.count_negated(stripped_tokens)
        feature_dict['emoticon_count'] = self.count_emoticons(emoticon_list)
        n_grams = self.find_bi_grams(stripped_tokens)
        
        tweet_data['n-grams'] = n_grams
        tweet_data['engineered features'] = feature_dict

        return tweet_data
    
    def vocabulary_updater(self, tweet_tokens):
        for token in tweet_tokens:
            if token not in self.vocabulary:
                self.vocabulary[token] = 1
            if token in self.vocabulary:
                self.vocabulary[token] += 1

    def infrequent_vocabulary_filter(self):
        self.vocabulary = {k:v for k,v in self.vocabulary.items() if v > 1}

    def feature_updating(self):
        count = 0
        with open(self.input_path, 'r') as f_in, open(self.output_path, 'w') as f_out:
            for line in f_in:
                tweet_data = json.loads(line)
                updated_tweet_data = self.feature_generator(tweet_data)

                # update vocabulary
                self.vocabulary_updater(updated_tweet_data['n-grams'])

                # write updated tweet_data to f_out
                f_out.write(json.dumps(updated_tweet_data) + "\n")
                count += 1
                if count % 1000 == 0:    
                    print(count)
        self.infrequent_vocabulary_filter()
        return list(self.vocabulary)

def load_vocabulary(voc_path):
    with open(voc_path, 'r') as f:
        vocabulary = json.load(f)
    return vocabulary

def write_vocabulary(voc_path, vocabulary):
    with open(voc_path, 'w') as f:
        json.dump(vocabulary, f)

def read_line(line):
    tweet_data = json.loads(line)
    tokens = tweet_data['n-grams']
    label = tweet_data['label']
    feature_dictionary = tweet_data['engineered features']
    return tokens, label, feature_dictionary

def append_sparse_matrix(new_sparse_vec, sparse_matrix):
    vector_to_stack = new_sparse_vec.T
    sparse_matrix = sparse.hstack((sparse_matrix, vector_to_stack))
    return sparse_matrix

def track_existing_features(feature_dict, existing_features=[]):
    for key in feature_dict:
        if feature_dict[key] != 0:
            if key not in existing_features:
                existing_features.append(key)
    return existing_features

def build_sparse_matrix(tweet_path, vocabulary):
    tweet2instance = features2Instance(vocabulary)
    labels = []
    with open(tweet_path, 'r') as f:
        
        # read the first line
        line = f.readline()
        tokens, label, feature_dictionary = read_line(line)
        existing_features = track_existing_features(feature_dictionary)
        sparse_matrix = tweet2instance.get_sparse_matrix_from_tweet(tokens, feature_dictionary).T
        labels.append(label)
        count = 0
        
        # skip the first tweet
        while True:
            line = f.readline()
            if not line:
                break
            tokens, label, feature_dictionary = read_line(line)
            existing_features = track_existing_features(feature_dictionary, existing_features)
            new_sparse_matrix = tweet2instance.get_sparse_matrix_from_tweet(tokens, feature_dictionary)
            sparse_matrix = append_sparse_matrix(
                                new_sparse_matrix, 
                                sparse_matrix
            )
            labels.append(label)
            count += 1
            if count % 100 == 0:   
                print(count)
                print(sparse_matrix.shape)

    return sparse_matrix, labels
    
def string_compiler(sparse_indices):
    string_to_write = ""
    for index, frequency in sparse_indices.items():
        string_to_write += f"{int(index)}:{frequency} "
    return string_to_write

def write_data_set(output_path, sparse_matrix, labels):
    with open(output_path, 'w') as f:
        for i in range(sparse_matrix.shape[1]):
            
            # TODO: there must be an off the shelf way to do this
            # so change to better code
            sparse_col = sparse_matrix.getcol(i)
            sparse_indices = sparse_col.nonzero()[0]
            sparse_values = [float(value) for value in sparse_col.data]
            # map the sparse index to its frequency in the tweet
            sparse_index2freq = dict(zip(sparse_indices, sparse_values))
            string_to_write = string_compiler(sparse_index2freq).strip(" ")
            f.write(f"{str(labels[i])} " + string_to_write + "\n")

            if i % 1000 == 0:
                print(i)