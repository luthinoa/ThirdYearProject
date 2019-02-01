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


def contains_number(inputString):

    return any(char.isdigit() for char in inputString)


service_path = os.path.abspath("service/nli-service-cli.py")
model_path = os.path.abspath("saved/snli/esim/2/esim")
model_real_path = model_path.replace('/bin','')


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

	#entailement -> should have been 
		#contradiction -> (what it is)
			#word count 
		#unknown -> (what it is)
	#what class should it be 
	#what class it was. 


#take top 3 words 
#dictionary between word and count of positive and negative 
#when you have 3 classes, for each class, right/wrong(for model prediction), and then check for bias in specific words. 



class_names = ['entailment','contradiction','neutral']
URL = "http://0.0.0.0:9001/nnli"
FILE_NAME = 'fever_labeled.jsonl'

with open(FILE_NAME) as f:
    labeled_fever_data = list(f)

print("LENGTH OF LABELED "+str(len(labeled_fever_data)))

data_with_numbers=[]
res = []


def get_model_prediction(arr): 

	url = "http://0.0.0.0:9001/nnli"

	ret = []
	for each in arr:
		data = {
			"sentence1": each[0],
			"sentence2": each[1]
		}	
		res=requests.post(url,data=data)
		res_json= res.json()
		predicted_label = get_predicted_label(res_json)
		print("Prediction: "+str(predicted_label))
		return get_predicted_label(res_json)


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


def call_service(arr):

	url = "http://0.0.0.0:9001/nnli"

	ret = []
	for each in arr:
		data = {
			"sentence1": each[0],
			"sentence2": each[1]
		}	
		res=requests.post(url,data=data)
		res_json= res.json()
		sentences = each[0]+" "+each[1]
		ret.append([float(res_json["contradiction"]), float(res_json["entailment"]), float(res_json["neutral"])])	
	
	return np.array(ret)


def call_lime(instance, isBow, num_of_features):

	explanations = []
	print(instance)
	explainer = LimeTextExplainer(class_names=class_names,bow=isBow)
	exp = explainer.explain_instance(instance,call_service, num_features=num_of_features)
	return exp.as_list()


for item in labeled_fever_data:
	instance = json.loads(item)

	if contains_number(instance["sentence1"]) or contains_number(instance["sentence2"]):
		string_instance = instance["sentence1"]+instance["sentence2"]
		instance["label"] = get_model_prediction([instance["sentence1"],instance["sentence2"]])
		instance["explanation"] = call_lime(string_instance,True,10)
		res.append(instance) 
		with open('result.txt', 'w') as outfile:  
			json.dump(instance, outfile)

		print("RESULT"+str(instance))
		break
		


