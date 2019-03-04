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


#run service with ./bin/service/nli-service-cli.py -R saved/snli/esim/2/esim -r 300 -m esim -p 9001
# python -m IPython notebook
DATA_FILENAME = 'explanations.jsonl'

def contains_number(inputString):

    return any(char.isdigit() for char in inputString)

service_path = os.path.abspath("service/nli-service-cli.py")
model_path = os.path.abspath("saved/snli/esim/2/esim")
model_real_path = model_path.replace('/bin','')

class_names = ['entailment','contradiction','neutral']
URL = "http://0.0.0.0:9001/nnli"
FILE_NAME = 'fever_labeled.jsonl'

with open(FILE_NAME) as f:
    labeled_fever_data = list(f)

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

def call_service(item):

	url = "http://0.0.0.0:9001/nnli"
	data = {
		"sentence1": item[0],
		"sentence2": item[1]
	}
	res=requests.post(url,data=data)
	res_json= res.json()
	return res_json

def get_model_prediction(data_item): 

	res_json= call_service(data_item)
	predicted_label = get_predicted_label(res_json)
	# print("Prediction: "+str(predicted_label))
	return get_predicted_label(res_json)


def call_service_lime(arr):

	ret = []
	for item in arr: 
		res_json=call_service(item)
		ret.append([float(res_json["contradiction"]), float(res_json["entailment"]), float(res_json["neutral"])])	
	
	return np.array(ret)

def call_lime(instance, isBow, num_of_features):

	explanations = []
	print(instance)
	explainer = LimeTextExplainer(class_names=class_names,bow=isBow)
	exp = explainer.explain_instance(instance,call_service_lime, num_features=num_of_features)
	return exp.as_list()

def main():

	list = []
	for item in labeled_fever_data:
		instance = json.loads(item)
		list.append(instance)

	sen1 = []
	sen2 = []

	print(len(list))

	for item in list: 
		sen1.append(item["sentence1"])
		sen2.append(item["sentence2"])

	print(len(sen1))
	print(len(sen2))


	data_items = {
		"sentence1":json.dumps(sen1),
		"sentence2":json.dumps(sen2)
	}

	
	# data = json.dumps(data_items)
	url = "http://0.0.0.0:9001/nnli"

	res=requests.post(url,data=data_items)
	# res_json= res.json()
	# return res_json

	# for item in labeled_fever_data:

	# 	instance = json.loads(item)

	# 	if contains_number(instance["sentence1"]) or contains_number(instance["sentence2"]):
	# 		string_instance = instance["sentence1"]+instance["sentence2"]
	# 		instance["label"] = get_model_prediction([instance["sentence1"],instance["sentence2"]])
	# 		instance["explanation"] = call_lime(string_instance,True,10)
	# 		res.append(instance)

	# with open(DATA_FILENAME,'w') as outfile:  
	# 	json.dump(res, outfile)

if __name__ == '__main__':
	main()



