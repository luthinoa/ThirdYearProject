import argparse
import untangle
import json
from collections import Counter

conv2fever={"entailment":"SUPPORTS", "contradiction":"REFUTES", "neutral":"NOT ENOUGH INFO"}

def read_conv(path):
    conv = read_jsonl(path)
    instances=dict()
    for pair in conv:
        instance=dict()
        instance["evidence"] = [pair["sentence1"]]
        instance["claim"] = pair["sentence2"]
        instance["id"] = pair["pairID"]
        instance["label"] = conv2fever[pair["gold_label"]]
        instances[pair["pairID"]]=instance
    return instances


rte2fever={"YES":"SUPPORTS", "NO":"REFUTES", "UNKNOWN":"NOT ENOUGH INFO"}

def read_rte(path):
    rte = untangle.parse(path)
    instances=dict()
    for pair in rte.entailment_corpus.pair:
        instance=dict()
        instance["evidence"] = [pair.t.cdata]
        instance["claim"] = pair.h.cdata
        instance["id"] = pair["id"]
        instance["label"] = rte2fever[pair["entailment"]]
        instances[pair["id"]]=instance
    return instances

def read_jsonl(path):
    with open(path, "r") as in_file:
        out = [json.loads(line) for line in in_file]

    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate predictions on gold labels")
    parser.add_argument(
        "in_file",
        help="input file path for rte (e.g., dev.sentences.p5.s5.jsonl)")
    parser.add_argument("save_preds", help="specify file name of saved predictions")
    args = parser.parse_args()
    gold = read_conv(args.in_file)
    preds = read_jsonl(args.save_preds)
    correct=0
    tot=0
    correcd=Counter()
    tod=Counter()
    correcp=Counter()
    top=Counter()
    for instance in preds:
        smax=0
        lmax=""
        for s,l in zip(instance["scores"][0],instance["predicted_labels"][0][0]):
            if s>smax:
                smax=s
                lmax=l
        if lmax==gold[instance["id"]]["label"]:
            correct=correct+1
            correcd[gold[instance["id"]]["label"]]+=1
            correcp[lmax]+=1
        tot=tot+1
        tod[gold[instance["id"]]["label"]]+=1
        top[lmax]+=1

print(correct,tot,correct/tot)
print()
for l,t in tod.most_common():
    print(l+":",correcd[l],t,correcd[l]/t)
print()
for l,t in top.most_common():
    print(l+":",correcp[l],t,correcp[l]/t)
    
