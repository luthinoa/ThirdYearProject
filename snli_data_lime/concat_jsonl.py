import json

DATA_OUTPUT_FILE = "snli_explanations.jsonl"

file_names = ["snli_0-2000_20190324-203411.jsonl","snli_2000-4000_20190324-204126.jsonl","snli_4000-6000_20190324-204919.jsonl","snli_6000-8000_20190324-205514.jsonl","snli_8000-10000_20190324-210143.jsonl"]



result = []

for file in file_names:
	print(file)
	with open(file) as f:
		lines = f.read().splitlines()
		result+=json.loads(lines[0])

with open(DATA_OUTPUT_FILE,'w') as outfile:  
		json.dump(result, outfile)

print(len(result))

  

