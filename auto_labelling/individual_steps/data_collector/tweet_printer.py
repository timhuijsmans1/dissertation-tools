import json
with open("./collected_data/filtered_search_results_07-18-2022_11;15;20.txt", 'r') as f:
	for tweet in f:
		text = json.loads(tweet)['text']
		print(text)
		print('-' * 30)
