import os 
import inspect 
import json
import requests
import numpy as np 
import lime
import re
from lime import lime_text 
from lime.lime_text import LimeTextExplainer

service_path = os.path.abspath("service/nli-service-cli.py")
model_path = os.path.abspath("saved/snli/esim/2/esim")
model_real_path = model_path.replace('/bin','')

class_names = ['entailment','contradiction','neutral']

sen1 = "The Ukrainian SSR was a founding member of the United Nations , although it was legally represented by the All-Union state in its affairs with countries outside of the Soviet Union"
sen2 = "Ukrainian Soviet Socialist Republic was a founding participant of the UN."
URL = "http://0.0.0.0:9001/nnli"


FILE_NAME = 'fever_labeled.jsonl'

with open(FILE_NAME) as f:
    labeled_fever_data = list(f)

print(len(labeled_fever_data))
arr = np.zeros((len(labeled_fever_data),2),dtype=object)

# for item in labeled_fever_data:
# 	inst = json.loads(item)

for i in range(0,len(labeled_fever_data)):
	instance = json.loads(labeled_fever_data[i])

	# json_instace = json_loads(str(instance))
	# print(str(instance["sentence1"]))
	# print(instance["sentence2"])
	# print(i)
	arr[i,0]=instance["sentence1"]
	arr[i,1]=instance["sentence2"]

#this function only takes numpy array 
#sentence 1 numpy array0 
#sentence 2 numpy array1 

#return numpy array of 3 size 3 (for each class)

# arr = np.array([sen1, sen2])
arr = np.array([[sen1, sen2]])

def call_service(arr):

	# result = np.zeros((len(labeled_fever_data),3))
	# data = {
	# 	"sentence1": arr[0][0],
	# 	"sentence2": arr[0][1]
	# }
	url = "http://0.0.0.0:9001/nnli"

	# for i in range(0,len(labeled_fever_data)):
	# 	data = {
	# 		"sentence1": labeled_fever_data[i][0],
	# 		"sentence2": labeled_fever_data[i][1]
	# 	}
	# 	print(data["sentence1"])
	ret = []
	for each in arr:
		data = {
			"sentence1": each[0],
			"sentence2": each[1]
		}	
		res=requests.post(url,data=data)
		res_json= res.json()
		ret.append([float(res_json["contradiction"]), float(res_json["entailment"]), float(res_json["neutral"])])
	res=requests.post(url,data=data)
	res_json= res.json()
	print(res_json)
	print((res_json["contradiction"]))

	# returned_scores = np.array([[float(res_json["contradiction"]), float(res_json["entailment"]), float(res_json["neutral"])]])
	
	return np.array(ret)

	# url = "http://0.0.0.0:9001/nnli"
	# res = requests.post(url, data=data)
	# res_json = res.json()
	# return np.array([[res_json["contradiction"], res_json["entailment"], res_json["neutral"]]])

# print(call_service(arr))
print(labeled_fever_data[0])
instance = sen1+sen2
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(instance,call_service,num_samples=10)
result = exp.as_list()
print(result)

#remove the url to global 
#
# call_service - is to make it a function that takes and returns a numpy array 
#make the two senteces a numpy array 
#convert the scores from numpy to scores 