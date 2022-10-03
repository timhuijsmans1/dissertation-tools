import os
import datetime
import json

from twarc import Twarc2, expansions
from itertools import islice

class TweetCollector():

    def __init__(self, 
                ticker_100_path,
                full_ticker_path, 
                nyse_path,
                emoticon_list, 
                client, 
                start_datetime, 
                end_datetime,
                output_path,
                full_nasdaq_list=True,
                nyse=True
        ):
        if full_nasdaq_list:
            self.ticker_list = self.ticker_loader_full(full_ticker_path, False)
            if nyse == True:
                self.ticker_list = self.add_nyse(nyse_path)   
        else:
            self.ticker_list = self.ticker_loader_100(ticker_100_path)
        
        # splits the ticker list into chunks because of the query size limit
        # of the tweet search API
        self.ticker_subsets = self.ticker_list_splitter()

        self.emoticon_list = emoticon_list
        self.client = client
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        self.output_path = output_path

    def ticker_loader_100(self, path):
        ticker_list = []
        with open(path, 'r') as f:

            # read all company lines
            while True:
                line = f.readline()
                if not line:
                    break
                # extract ticker symbol from company line
                ticker = line.split(',')[1]
                ticker = "$" + ticker
                ticker_list.append(ticker)
        
        return ticker_list

    def ticker_loader_full(self, path, nyse):
        ticker_list = []

        with open(path, 'r') as f:
            ticker_dict = json.load(f)
            # dataset formats provided differ for the two exchanges,
            # hence the different symbol keys
            if nyse:
                ticker_list = ["$" + company_dict['ACT Symbol'] for company_dict in ticker_dict]    
            if not nyse:
                ticker_list = ["$" + company_dict['Symbol'] for company_dict in ticker_dict]
        
        return ticker_list

    def add_nyse(self, path):
        nyse_tickers = set(self.ticker_loader_full(path, True))
        total_tickers = list(set(self.ticker_list) | nyse_tickers)
        return total_tickers

    def ticker_list_splitter(self):
        iterator = iter(self.ticker_list)
        
        return iter(lambda: tuple(islice(iterator, 75)), ())
    
    def combined_query_compiler(self, ticker_list_subset):
        # compile lists of tickers and emoticons with OR bool
        ticker_query = "(" + " OR ".join(ticker_list_subset) + ")"
        emoticon_query = "(" + " OR ".join(self.emoticon_list) + ")"

        # combine tickers, emoticons and query requirements
        query = ticker_query + " " + emoticon_query + " -is:retweet lang:en"
    
        return query

    def ticker_query_compiler(self, ticker_list):
        # compile lists of tickers and emoticons with OR bool
        ticker_query = "(" + " OR ".join(ticker_list) + ")"

        # combine tickers, emoticons and query requirements
        query = ticker_query + " -is:retweet lang:en"
    
        return query

    def execute_search(self, query):
        search_results = self.client.search_all(
                            query=query, 
                            start_time=self.start_datetime, 
                            end_time=self.end_datetime, 
                            max_results=100
        )

        return search_results

    def result_filter(self, tweet, ticker_filter=False):
        emoticon_bool = False
        ticker_bool = False
        
        tweet_text = tweet["text"]

        # check if the tweet indeed contains a nasdaq 100 ticker or an emoticon
        for emoticon in self.emoticon_list:
            if emoticon in tweet_text:
                emoticon_bool = True
                break
        # default is to not filter by ticker because of capitals 
        if not ticker_filter:
            return emoticon_bool

        # if filter ticker is enabled, we make sure that the tweet indeed
        # contains one of the tickers in our ticker_list
        if ticker_filter:
            for ticker in self.ticker_list:
                if ticker in tweet_text:
                    ticker_bool = True
                    break

            return ticker_bool * emoticon_bool

    def result_writer(self, query_list):

        with open(self.output_path, 'w+') as f: 
            # execute the search for every subset of the ticker list
            for i, query in enumerate(query_list):
                print(f'processing results of {i+1}/{len(query_list)}')
                search_results = self.execute_search(query)
                
                # this is required because of the result generator returned
                # by the Twitter API
                for page in search_results:
                    result = expansions.flatten(page)

                    # check if tweets are valid and write valid tweets
                    for tweet in result:
                        f.write('%s\n' % json.dumps(tweet))

        return

    def execute_combined_search(self):
        "executes the search for both companies and emoticons combined"

        query_list = [self.combined_query_compiler(tickers) for tickers in self.ticker_subsets]
        self.result_writer(query_list)

    def execute_company_search(self, ticker_list):
        "executes the search for companies only, omitting the emoticon filter"

        query_list = [self.ticker_query_compiler(ticker_list)]
        self.result_writer(query_list)

if __name__ == "__main__":
    # global vars
    NASDAQ_100_PATH = "./nasdaq_100_listings.csv"
    FULL_NASDAQ_PATH = "./nasdaq-listed-symbols.json"
    NYSE_PATH = "./nyse-listed.json"
    NOW = datetime.datetime.now().strftime("%m-%d-%Y_%H;%M;%S")
    OUTPUT_PATH = f"./collected_data/filtered_search_results_{NOW}.txt"
    BEARER = os.environ.get("BEARER")
    CLIENT = Twarc2(bearer_token=BEARER)
    # START_TIME = datetime.datetime(2017, 11, 9, 0, 0, 0, 0, datetime.timezone.utc)
    # END_TIME = datetime.datetime(2022, 5, 30, 0, 0, 0, 0, datetime.timezone.utc)
    START_TIME = datetime.datetime(2022, 7, 17, 0, 0, 0, 0, datetime.timezone.utc)
    END_TIME = datetime.datetime(2022, 7, 18, 0, 0, 0, 0, datetime.timezone.utc)
    POS_EMOTICON_LIST = ["😀", "😃", "😄", "😁", "🙂"]
    NEG_EMOTICON_LIST = ["😡", "😤", "😟", "😰", "😨", "😖", "😩", "🤬", "😠", "💀", "👎", "📉"]
    EMOTICON_LIST = POS_EMOTICON_LIST + NEG_EMOTICON_LIST
    TICKER_LIST = ['$AAPL']

    # execute program
    tweet_collector = TweetCollector(
                        NASDAQ_100_PATH,
                        FULL_NASDAQ_PATH,
                        NYSE_PATH,
                        EMOTICON_LIST,
                        CLIENT,
                        START_TIME,
                        END_TIME,
                        OUTPUT_PATH
    )
    # tweet_collector.execute_combined_search()
    tweet_collector.execute_company_search(TICKER_LIST)