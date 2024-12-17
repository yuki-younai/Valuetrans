import json
import argparse
from src.model.api import LLM_API
from src.model.llm import LLM_local
from src.role_person_generation import generate_persona_description
from src.role_occupation_generation import generate_persona_occupation_description
from src.evaluate import evaluate_mfq30 , evaluation_pvqrr
import re
import json
from collections import Counter
import os
from utils import get_circle_list
API_MODEL = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]
from tqdm import tqdm
from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "token-abc123"


def init_dataset(datasets_name):
    
    if datasets_name=="mfq30":
        with open("Datasets/mfq-30/test2.json", 'r') as file:
            prompt_data = json.load(file)
    elif datasets_name=="pvqrr":
        with open("Datasets/pvq-rr/test2_new.json", 'r') as file:
            prompt_data = json.load(file)

    return prompt_data

def extract_answer(api_model, origin_response, response, options, origin_ques, circle=True):
    results = []
    letters = []
    if circle:
        new_options = get_circle_list(options)
    else:
        new_options = [options]
    i = 0
    letter2int = {
        'A':0,
        'B':1,
        'C':2,
        'D':3,
        'E':4,
        'F':5
    }
    for op in new_options:
        extract_answer_example =  f"""
        Question: {origin_ques}
        There is a person X whose answer is {origin_response}.
        There is a person Y whose answer is {response}.
        Disregarding the fact that X has not provided a detailed explanation, please assess the similarity between Y's and X's responses. Below are the options for similarity:
        Use the above information to determine which option B's response is more consistent with A's response.
        Options: A. {op[0]} B. {op[1]} C. {op[2]} D. {op[3]} E. {op[4]} F. {op[5]}
        Please give me the choice letter first and then give me the reason.
        """
        messages = []
        messages.append({"role":"user", "content": extract_answer_example})
        answer = completion(model="openai/Qwen2.5-32B-Instruct", api_base="http://localhost:8811/v1", messages=messages, temperature=0).choices[0].message.content
        results.append(answer)
        if answer[0] in letter2int.keys():
            letters.append((letter2int[answer[0]]+i)%6)
        else:
            letters.append('')
        i+=1
    import pdb
    pdb.set_trace()
    return answer


def single_run(model, prompt_data, personas, dataset_name):
    for i, inst in enumerate(prompt_data):
        inst['prompt{}'.format(i)]={}
        inst['prompt{}'.format(i)]['response'] = []
        inst['prompt{}'.format(i)]['prompt'] = inst['question']
        print("##############################",i,"/",len(prompt_data),"###########################################")
        answer_list = []
        for p_idx, per in tqdm(enumerate(personas), desc="Processing Personas", ncols=10):
            # print("----------------------------------",p_idx,"/",len(personas),"-----------------------------------")

            prompt = "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
            prompt += f"{per['description']} \n\n"
            prompt += "Use the given information to answer the question below. \n\n"
            prompt += inst['question']
            messages = []
            messages.append({"role":"user", "content": prompt})
            model_response = model.respond(messages)
            if p_idx % 10 == 0:
                print(model_response)
                print('='*50)
            # answer = extract_answer(help_model, model_response, inst['options'], inst["centerpiece"])
            # answer = extract_answer(help_model, inst['response'], model_response, inst['options'], inst['question'])
        #     if answer!="NONE":  
        #         answer_list.append(answer[0])
        #     print("Respond:",model_response)
        #     print("Extract Answer:", answer)
        # if len(answer_list)==0:
        #     answer_list.append("A")
        # frequency = Counter(answer_list)    
        # most_frequent_element = frequency.most_common(1)[0][0]
            inst['prompt{}'.format(i)]['response'].append(model_response)
        # inst['respond_answer'] = most_frequent_element
        # inst['respond_list'] = answer_list

    return prompt_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #1.基本选择：选择模型，数据集，模型配置，训练方式
    parser.add_argument("--dataset", type=str, default="mfq30",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--output_dir", type=str, default="output/exp1",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--model_output", type=str, default="output/exp1",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--role", type=str, default="place",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--run", type=str, default="single",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--run_results", type=str, default=None,
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--gpu", type=int, default=3,
                    help="Random seed.")
    parser.add_argument("--role_num", type=int, default=2,
                    help="Random seed.")
    batch_size = 1
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open("src/config.json", 'r') as file:
        args.config = json.load(file)

    prompt_data = init_dataset(args.dataset)


    help_model =  LLM_API("gpt-3.5-turbo", api_key=args.config['openai_api_key'], base_url=args.config['base_url'])


    if args.role=="base":
        personas = generate_persona_description(args.role_num)
    elif args.role=="place":
        personas = generate_persona_occupation_description(args.role_num)


    if not os.path.exists(args.model_output):
        if args.model in API_MODEL:
            model = LLM_API(args.model, api_key=args.config['openai_api_key'], base_url=args.config['base_url'])
        else:
            model = LLM_local(args.model, args.gpu)
        prompt_data = single_run(model, prompt_data, personas, args.dataset)
        with open(args.output_dir+"/run_result.json", 'w') as file:
            json.dump(prompt_data, file, indent=4)
        args.model_output = args.output_dir+"/run_result.json"

    # with open(args.model_output, 'r') as file:
    #     prompt_data = json.load(file)


    # if args.dataset=="mfq30":
    #     logs_info = """
    #     Evaluation Log
    #     --------------
    #     Evaluation Results:
    #     - Harm: {harm}
    #     - Fairness: {fairness}
    #     - Ingroup: {ingroup}
    #     - Authority: {authority}
    #     - Purity: {purity}
    #     """
    #     param_dict = evaluate_mfq30(prompt_data)
    # elif args.dataset=="pvqrr":
    #     logs_info = """
    #     Evaluation Log
    #     --------------
    #     Evaluation Results:
    #     - Self-Direction: {Self_Direction}
    #     - Stimulation: {Stimulation}
    #     - Hedonism: {Hedonism}
    #     - Achievement: {Achievement}
    #     - Power: {Power}
    #     - Security: {Security}
    #     - Conformity: {Conformity}
    #     - Tradition: {Tradition}
    #     - Benevolence: {Benevolence}
    #     - Universalism: {Universalism}
    #     """
    #     param_dict = evaluation_pvqrr(prompt_data)

    # logs_info = logs_info.format(**param_dict)
    # # 写入文件
    # with open(args.output_dir+'/logs.txt', 'a', encoding='utf-8') as file:
    #     file.write(logs_info)





