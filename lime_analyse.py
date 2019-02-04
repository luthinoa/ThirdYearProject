import json

def count_words(explanation, d, num_of_top_features):

	top_words = get_top_features(explanation,num_of_top_features)
	print("TOPWORDS "+str(top_words))
	for word_feature in top_words:
		if word_feature[0] in d:
			d[word_feature[0]]+=1
		else:
			d[word_feature[0]] = 1
	return d 

#this functions sorts the explanation according to absolute weights and returns a list of the top three weighted words
def get_top_features(explanation, num_of_top_features):

	if len(explanation) <= num_of_top_features:
		return explanation
	
	explanation.sort(key=lambda row: abs(row[1]), reverse=True)

	return explanation[:num_of_top_features]

def main():

	FILE_NAME = 'result.jsonl'
	explained_fever_data = []

	with open(FILE_NAME) as f:
	    explained_fever_data = list(f)
	
	print(explained_fever_data)

	for instance in explained_fever_data:

		instance = json.loads(instance)

		gold_label = instance['gold_label']
		predicted_label = instance['label']

		if gold_label == predicted_label:
			continue

		res = {
			"neutral": {
				"neutral": {},
				"contradiction": {},
				"entailment": {}
			},
			"contradiction": {
				"neutral": {},
				"contradiction": {},
				"entailment": {}
			},
			"entailment": {
				"neutral": {},
				"contradiction": {},
				"entailment": {}
			}
		}

		explanation = instance['explanation']

		count_words(explanation, res[gold_label][predicted_label], 3)


	print(res)


if __name__ == '__main__':
	main()


#take top 3 words 
#dictionary between word and count of positive and negative 
#when you have 3 classes, for each class, right/wrong(for model prediction), and then check for bias in specific words. 


