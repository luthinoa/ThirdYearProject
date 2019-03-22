import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import time
from sklearn.metrics import mean_squared_error, r2_score



#add timestamp to filename 
#get proper plot bars of word count to pdf
#heat map 
#ask pasquale if it's worth only changing sen1 and sen2 instead of both. 

#for loop for json 
#concatenate the sent and send to lime
#inside lime we break 
#instead - give him only sentence1/sentence2 

timestr = time.strftime("%Y%m%d-%H%M%S")
FILE_NAME = 'numbers_BOW_1000INST_1000SAMP_10FEAT.jsonl'
DATA_OUTPUT_FILE = FILE_NAME+'_count_words'+timestr
pdf = PdfPages(timestr+"_"+DATA_OUTPUT_FILE+".pdf")


def count_words(explanation, d, num_of_top_features):

	top_words = get_top_features(explanation,num_of_top_features)
	for word_feature in top_words:
		if word_feature[0] in d:
			d[word_feature[0]]+=1
		else:
			d[word_feature[0]] = 1
	return d


def filter_word_count(dict_data,threshold):

	labels = dict_data.keys() #neutral,cont,ent
	for label in labels:
		sub_labels = dict_data[label].keys()  #neutral,cont,ent
		for inner_label in sub_labels:
			words = dict_data[label][inner_label].keys() #all words in word count
			for word in words:
				count = dict_data[label][inner_label][word]
				if(count<=threshold):
					inner_dict = dict_data[label][inner_label]
					dict_data[label][inner_label] = remove_from_dict(inner_dict,word)

	return dict_data

def remove_from_dict(dict_data,key):
    r = dict(dict_data)
    del r[key]
    return r

#this functions sorts the explanation according to absolute weights and returns a list of the top three weighted words
def get_top_features(explanation, num_of_top_features):

	if len(explanation) <= num_of_top_features:
		return explanation
	
	explanation.sort(key=lambda row: abs(row[1]), reverse=True)

	return explanation[:num_of_top_features]

def plot_word_count_bars(dict_data):

	key_set = dict_data.keys()

	for key in key_set:
		df = pd.DataFrame(dict_data[key])
		fig = df.plot(kind="bar").get_figure()
		fig.suptitle(key)
		pdf.savefig(fig)

	pdf.close()

def plot_explanations(data_instances):


	

def main():

	explained_fever_data = []

	with open(FILE_NAME) as json_file:
	    explained_fever_data = json.load(json_file)

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
	
	for instance in explained_fever_data:

		gold_label = instance['gold_label']
		predicted_label = instance['label']

		explanation = instance['explanation']

		count_words(explanation, res[gold_label][predicted_label], 3)

	print("neutral neutral: "+str(len(res['neutral']['neutral'])))
	print("neutral contradiction: "+str(len(res['neutral']['contradiction'])))
	print("neutral entailment: "+str(len(res['neutral']['entailment'])))
	print("contradiction neutral: "+str(len(res['contradiction']['neutral'])))
	print("contradiction contradiction: "+str(len(res['contradiction']['contradiction'])))
	print("contradiction entailment: "+str(len(res['contradiction']['entailment'])))
	print("entailment neutral: "+str(len(res['entailment']['neutral'])))
	print("entailment contradiction: "+str(len(res['entailment']['contradiction'])))
	print("entailment entailment: "+str(len(res['entailment']['entailment'])))

	filter_word_count(res,3)
	plot_word_count_bars(res)

	with open(DATA_OUTPUT_FILE,'w') as outfile:  
		json.dump(res, outfile)


if __name__ == '__main__':
	main()


#take top 3 words 
#dictionary between word and count of positive and negative 
#when you have 3 classes, for each class, right/wrong(for model prediction), and then check for bias in specific words. 


