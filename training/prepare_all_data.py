import os
import pandas as pd
import json
from pdb import set_trace

dataset_path= "/data2/szs/value/dataset/rewrite_500_qa"
output_path= "/data2/szs/value/dataset/train_dataset"
jsons = os.listdir(dataset_path)
rejected = "all"
chosen_scores = {
    "Achievement":5,"Security":1,"Stimulation":1, "Self":1, "Conformity":1, "Hedonism":5, "Benevolence":1, "Power":5
}
rejected_scores = {
    "Achievement":1,"Security":5,"Stimulation":5, "Self":5, "Conformity":5, "Hedonism":1, "Benevolence":5, "Power":1
}
raw_dataset={}

for fi in jsons:
    if '.json' in fi:
        with open(os.path.join(dataset_path, fi), 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw_dataset[fi.split('_')[0]] = data

dataset={
    'chosen':[],
    'rejected':[]
}

for k in raw_dataset:
    if rejected == "all":
        rejected_list = [1,2,3,4,5]
        rejected_list.remove(chosen_scores[k])
    else:
        rejected_list = [rejected_scores[k]]
        
    for i in range(len(raw_dataset[k])):
        for r in rejected_list:
            dataset['chosen'].append([
                {
                    'content':raw_dataset[k][i]['prompt'], 'role':'user'
                },
                {
                    'content':raw_dataset[k][i][f'response_{chosen_scores[k]}'], 'role':'assistant'
                },
            ])
            dataset['rejected'].append([
                {
                    'content':raw_dataset[k][i]['prompt'], 'role':'user'
                },
                {
                    'content':raw_dataset[k][i][f'response_{r}'], 'role':'assistant'
                },
            ])
# set_trace()

dataset = pd.DataFrame(dataset)
if rejected=='all':
    dataset.to_parquet(os.path.join(output_path, '-'.join([k+'_'+str(chosen_scores[k]) for k in chosen_scores])+'_all.parquet'), index=False)
else:
    dataset.to_parquet(os.path.join(output_path, '-'.join([k+'_'+str(chosen_scores[k])+'_'+str(rejected_scores[k]) for k in chosen_scores])+'_all.parquet'), index=False)

