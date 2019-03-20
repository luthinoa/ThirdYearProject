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
import time

#run service with ./bin/service/nli-service-cli.py -R saved/snli/esim/2/esim -r 300 -m esim -p 9001
# python -m IPython notebook
# copy files from remote to local:
# scp nluthi@vic.cs.ucl.ac.uk:/home/nluthi/dev/ThirdYearProject/ numbers_H+P_1000INST_1000SAMP_10FEAT.jsonl /Users/noaluthi/dev/ThirdYearProject

timestr = time.strftime("%Y%m%d-%H%M%S")
DATA_FILENAME = 'fever_labeled.jsonl'
DATA_OUTPUT_FILE = explanations+"_"+timestr+'.jsonl'
class_names = ['entailment','contradiction','neutral']
URL = "http://0.0.0.0:9001/nnli"

def contains_number(inputString):

    return any(char.isdigit() for char in inputString)

#get data from labeled data json file, create a list of object which contains 
#sentence1 and sentence 2 for each object. 
def get_data(file_name):
	labeled_fever_data = []

	with open(DATA_FILENAME) as f:
		labeled_fever_data = list(f)
	string_json = json.dumps(labeled_fever_data)
	json_data = json.loads(string_json)

	all_instances = []

	for item in json_data:

		sentence1 = json.loads(item)["sentence1"]
		sentence2 = json.loads(item)["sentence2"]
		gold_label = json.loads(item)["gold_label"]

		data_item = {
			"sentence1": sentence1,
			"sentence2": sentence2,
			"gold_label": gold_label
		}
		all_instances.append(data_item)

	return all_instances

def get_data_in_range(file_name,start,end):

	data_list = get_data(file_name)
	res = []
	for i in range(start,end):
		res.append(data_list[i])
	return res 


# this will input arr - all augmented versions of the instance
# split two sentence according to "LIME_SPLIT" expression
# produce labels for them 
# and output as np.arr 

def call_service(data_items):

	sen1 = []
	sen2 = []

	for item in data_items:
		sentence1 = item["sentence1"]
		sentence2 = item["sentence2"]

		sen1.append(sentence1)
		sen2.append(sentence2)

	post_data = {
		"sentence1":json.dumps(sen1),
		"sentence2":json.dumps(sen2)
	}
	res = requests.post(URL,data=post_data)

	try:
		res_json = res.json()
		return res_json

	except Exception as e: 
		print(e)
		pass



def call_service_lime(arr):
	sen1 = []
	sen2 = []

	for item in arr: 
		try: 		
			sentence1, sentence2 = item.split("LIME_SPLIT")
			sen1.append(sentence1)
			sen2.append(sentence2)
		except Exception as e:
			print(e)
			pass

	data_items = {
		"sentence1":json.dumps(sen1),
		"sentence2":json.dumps(sen2)
	}

	res=requests.post(URL,data=data_items)
	res_json = res.json()

	ret = []
	for item in res_json:
		ret.append([float(item["contradiction"]), float(item["entailment"]), float(item["neutral"])])	

	return np.array(ret)



def call_lime(instance, isBow, num_of_features):
	instance = instance["sentence1"]+"LIME_SPLIT"+instance["sentence2"]
	explainer = LimeTextExplainer(class_names=class_names,bow=isBow,split_expression=u'\W+|LIME_SPLIT')
	exp = explainer.explain_instance(instance,call_service_lime, num_features=num_of_features, num_samples=100)
	return exp.as_list()


# classifier_fn – classifier prediction probability function, which takes a list of d strings and outputs a (d, k) 
# numpy array with prediction probabilities,
# where k is the number of classes. For ScikitClassifiers , this is classifier.predict_proba.


def get_predicted_label(result_json):

	contradiction = float(result_json["contradiction"])
	entailment = float(result_json["entailment"])
	neutral = float(result_json["neutral"])

	result_class = "contradiction"
	Max = contradiction

	if entailment > Max:
		Max = entailment
		result_class = "entailment"
	if neutral > Max:
		Max = neutral
		result_class = "neutral"
		if entailment > neutral:
			Max = entailment
			result_class = "entailment"

	return result_class

def append_model_predictions(data_items):

	res_json = call_service(data_items)
	for i in range(len(res_json)):
		data_items[i]["label"]=get_predicted_label(res_json[i])

	return data_items

#iterate on data instances, produce LIME explanation for each instance. 
def append_lime_explanations(data_instances,isBow,num_of_features):

	for item in data_instances:
		item["explanation"] = call_lime(item,isBow, num_of_features)

	return data_instances

def get_data_with_numbers(data_instances):

	data_with_numbers = []
	for item in data_instances:
		if(contains_number(item["sentence1"]) or contains_number(item["sentence2"])):
			data_with_numbers.append(item)

	return data_with_numbers


def main():	
	data_size = 28840
	half = 14420
	data_instances = get_data_in_range(DATA_FILENAME,0,2000)
	data_instances = get_data_with_numbers(data_instances)
	data_with_labels = append_model_predictions(data_instances)
	data_with_explanations = append_lime_explanations(data_with_labels,True,5)


	with open(DATA_OUTPUT_FILE,'w') as outfile:  
		json.dump(data_with_explanations, outfile)

if __name__ == '__main__':
	main()



