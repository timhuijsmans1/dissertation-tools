# Auto Labelling Tool

This folder contains all the scripts and the main file running the individual scripts to collect, pre-process, and label Tweets and turn the labelled Tweets into instance files ready for prevalence prediction. The individual scripts can be ran to execute a single step, or the combined process can be ran to execute all steps automatically.

## Usage
### IMPORTANT: In order to run the scripts, a Twitter v2 academic access bearer token is required.
To execute the collection pipeline, first place the bearer token in traindata_labelling_pipeline.py. Then, run the script as python traindata_labelling_pipeline.py and provide the date range in the command line prompt. The script executes if there is a stable internet connection, and the internet connection is required throughout the collection step (shown in command line output).

The outputs produced are the collected, processed and labelled data files, as well as the input that is required for the ML prediction process in the labelled_data2predictions folder.

