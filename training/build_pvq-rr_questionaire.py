import json
import argparse
from pdb import set_trace
import random
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#1.基本选择：
parser.add_argument("--person_json", type=str, default="../Datasets/person_500.json",
                    help="")
parser.add_argument("--person_num", type=int, default=20,
                    help="")
parser.add_argument("--if_circle", type=bool, default=True,
                    help="")
parser.add_argument("--output_path", type=str, default="../Datasets/test.parquet",
                    help="")
args = parser.parse_args()

with open(args.person_json, 'r') as file:
    person_json = json.load(file)

persons = random.sample(person_json, args.person_num)

with open("../Datasets/pvq-rr/test2_new.json", 'r') as file:
    question_response = json.load(file)

new_dataset = {
    'question_id':[],
    'question':[],
    'origin_response':[],
    'person_infor':[]
}

for qr in question_response:
    for person in persons:
        new_dataset['question_id'].append(qr['question_id'])
        new_dataset['question'].append(qr['question'])
        new_dataset['origin_response'].append(qr['response'])
        new_dataset['person_infor'].append(person['description'])

df = pd.DataFrame(new_dataset)
df.to_parquet(args.output_path)


    
    