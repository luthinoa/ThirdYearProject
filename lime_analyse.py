import os 
import inspect 
import json
import requests
import numpy as np 
import lime
import re
from lime import lime_text 
from IPython import get_ipython
from lime.lime_text import LimeTextExplainer

FILE_NAME = 'result.jsonl'
explained_fever_data = []

with open(FILE_NAME) as f:
    explained_fever_data = list(f)

#get all labeled instances 
#for instance in instances 
	#exp1 = explainer(H+P)
	#exp2 = explainer(P+H)
	#exp3 = explainer(BoW) look into 3 different BoWs 
	#visualize each explanation!!! 
	#create a numpy array of explanations. for each instance 3 dimensions, 1 for each explanation. 

#take top scores (i.e 3) for each 10 feature explanation. 
#for each word, count it 

#dictionary for each class

	#entailment -> should have been 
		#contradiction -> (what it is)
			#word count 
		#unknown -> (what it is)
	#what class should it be 
	#what class it was. 


#we generate three nested dictionaries, One for each class. 
#outer layer of dictionaries is the expected label for the instance. 
#inner layer of dictionaries is the model's prediction, which holds a word count 
entailment = {}
entailment['contradiction'] = {}
entailment['neutral'] = {}

contradiction = {}
contradiction['entailment'] = {}
contradiction['neutral'] = {}

neutral = {}
neutral['entailment'] = {}
neutral['contradiction'] = {}


def count_words(explanation, dict,num_of_top_features):

	top_words = get_top_features(explanation,num_of_top_features)
	print("TOPWORDS "+str(top_words))
	for word_feature in top_words:
		dict[word_feature[0]]=+1
	return dict 

#this functions sorts the explanation according to absolute weights and returns a list of the top three weighted words
def get_top_features(explanation,num_of_top_features):

	col = 1 
	top_words = []
	sorted_explanation = sorted(explanation, key=lambda row: np.abs(row[col]))
	print("SORTED"+ str(sorted_explanation))
	num_of_features = len(sorted_explanation)-1

	i=0
	while(i<num_of_top_features):
		# print("ELEEMT "+str(sorted_explanation[num_of_features-i]))
		top_words.append(sorted_explanation[num_of_features-i])
		i=i+1

	return np.array(top_words)


print(explained_fever_data)

for instance in explained_fever_data:

	instance = json.loads(instance)

	gold_label = instance['gold_label']
	predicted_label = instance['label']

	if(gold_label==predicted_label):
		continue

	explanation = instance['explanation']

	if (gold_label=='entailment'):
		count_words(explanation,entailment[str(predicted_label)],3)
	elif(gold_label=='neutral'):
		count_words(explanation,neutral[str(predicted_label)],3)
	elif(gold_label=='contradiction'):
		print(predicted_label)
		count_words(explanation,contradiction[srt(predicted_label)],3)


print(entailment['contradiction'])
print(entailment['neutral'])
print(neutral['entailment'])
print(neutral['contradiction'])
print(contradiction['entailment'])
print(contradiction['neutral'])



#take top 3 words 
#dictionary between word and count of positive and negative 
#when you have 3 classes, for each class, right/wrong(for model prediction), and then check for bias in specific words. 


