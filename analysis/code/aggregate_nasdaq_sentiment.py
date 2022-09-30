from single_ticker_prevalence import single_file_path_loader, single_stock_correlations

if __name__ == "__main__":
    TICKER_PARENT_PATH = "../../labelled_data2predictions/data/aggregate_sentiment_output"
    daily_file_paths = single_file_path_loader(TICKER_PARENT_PATH, '^NDX')
    single_stock_correlations(daily_file_paths, '^NDX')