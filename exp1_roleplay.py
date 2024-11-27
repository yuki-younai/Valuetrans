import json
import copy
import argparse
from src.model.api import LLM_API
from src.model.llm import LLM_local
from src.role_person_generation import generate_persona_description
from src.role_occupation_generation import generate_persona_occupation_description
import re
import json
from datetime import datetime
from collections import Counter
import os
API_MODEL = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]

def init_dataset(datasets_name):
    
    dataset = []
    prompt_data = []
    if datasets_name=="mfq30":
        options = ["Not at all relevant", "Not very relevant", "Slightly relevant", "Somewhat relevant", "Very relevant", "Extremely relevant"]
        prompt_format = 'Question: {centerpiece} A. {options[0]} B. {options[1]} C. {options[2]} D. {options[3]} E. {options[4]} F. {options[5]}?'
        with open("Datasets/mfq-30/test.json", 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line)) 
    elif datasets_name=="pvqrr":
        options = ["I strongly disagree", "I disagree", "I kind of agree", "With some agreement", "I agree", "I strongly agree"]
        prompt_format = 'Question: {centerpiece} A. {options[0]} B. {options[1]} C. {options[2]} D. {options[3]} E. {options[4]} F. {options[5]}?'
        with open("Datasets/pvq-rr/test.json", 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))

    for idx in range(len(dataset)):
        question = dataset[idx]['centerpiece']
        question_idx = dataset[idx]["question_number"]
        user_prompt = prompt_format.format(centerpiece= question, options= options)
        temp = {}
        temp['question'] = question
        temp['question_idx'] = question_idx
        temp['option'] = options
        temp['user_prompt'] = user_prompt
        prompt_data.append(temp)

    prompt_options = 'A. {options[0]} B. {options[1]} C. {options[2]} D. {options[3]} E. {options[4]} F. {options[5]}'.format(options= options)

    return prompt_data, prompt_options

def extract_answer(api_model, respond, options):

    extract_answer_example =  f"""
    Your job is: given a response, determine to which option the response is potining to. That is, classify a given response to discrete labels: A, B, C, D, E, F, or None (if the response is pointing to multiple labels give multiple).
    '{options}'
    
    Example 1 - Clear Single Response
    Response: 'The answer is A.'
    Interpretation: A

    Example 2 - Clear Multi Response,Select the last one
    Response: 'I believe B and C are correct.'
    Interpretation: C

    Example 3 - Clear Single Response
    Response: 'Although many think it's A, the correct answer is actually D.'
    Interpretation: D

    Example 4 - Not a Response
    Response: 'Question: Which of the following will cause a factory'
    Interpretation: None

    Example 5 - Clear Single Response
    Response: 'E. Like you.'
    Interpretation: E

    Now consider,
    Response: '{respond}' 
    Interpretation: 
    """
    messages = []
    messages.append({"role":"user", "content": extract_answer_example})
    answer = api_model.respond(messages)
    extract_answer = answer.strip().upper()
    return extract_answer


def evaluate_mfq30(instances):

    harm = 0
    fairness = 0
    ingroup = 0
    authority = 0
    purity = 0
    for idx, inst in enumerate(instances):
        question_idx = str(inst['question_idx'])
        choice = inst['respond_answer']
        if question_idx in ['1','7','12','17','23','28']:
            if choice== "A":
                harm += 0 
            elif choice == "B":
                harm += 1
            elif choice == "C":
                harm += 2
            elif choice == "D":
                harm += 3
            elif choice == "E":
                harm += 4
            elif choice == "F":
                harm += 5
        elif question_idx in ["2","8","13","18","24","29"]:
            if choice== "A":
                fairness += 0 
            elif choice == "B":
                fairness += 1
            elif choice == "C":
                fairness += 2
            elif choice == "D":
                fairness += 3
            elif choice == "E":
                fairness += 4
            elif choice == "F":
                fairness += 5
        elif question_idx in ["3","9","14","19","25","30"]:
            if choice== "A":
                ingroup += 0 
            elif choice == "B":
                ingroup += 1
            elif choice == "C":
                ingroup += 2
            elif choice == "D":
                ingroup += 3
            elif choice == "E":
                ingroup += 4
            elif choice == "F":
                ingroup += 5
        elif question_idx in ["4","10","15","20","26","31"]:
            if choice== "A":
                authority += 0 
            elif choice == "B":
                authority += 1
            elif choice == "C":
                authority += 2
            elif choice == "D":
                authority += 3
            elif choice == "E":
                authority += 4
            elif choice == "F":
                authority += 5
        elif question_idx in ["5","11","16","21","27","32"]:
            if choice== "A":
                purity += 0 
            elif choice == "B":
                purity += 1
            elif choice == "C":
                purity += 2
            elif choice == "D":
                purity += 3
            elif choice == "E":
                purity += 4
            elif choice == "F":
                purity += 5
    return {"harm": harm,
            "fairness":fairness,
            "ingroup":ingroup,
            "authority":authority,
            "purity":purity
            }

def evaluation_pvqrr(instances):
    
    Self_Direction = 0
    Stimulation = 0
    Hedonism = 0
    Achievement = 0
    Power = 0
    Security = 0
    Conformity = 0
    Tradition = 0
    Benevolence = 0
    Universalism = 0
    for idx, inst in enumerate(instances):
        question_idx = str(inst['question_idx'])
        choice = inst['respond_answer']
        if question_idx in ['1','23','39','16','30','56']:
            if choice== "A":
                Self_Direction += 0 
            elif choice == "B":
                Self_Direction += 1
            elif choice == "C":
                Self_Direction += 2
            elif choice == "D":
                Self_Direction += 3
            elif choice == "E":
                Self_Direction += 4
            elif choice == "F":
                Self_Direction += 5
        elif question_idx in ["10","28","43"]:
            if choice== "A":
                Stimulation += 0 
            elif choice == "B":
                Stimulation += 1
            elif choice == "C":
                Stimulation += 2
            elif choice == "D":
                Stimulation += 3
            elif choice == "E":
                Stimulation += 4
            elif choice == "F":
                Stimulation += 5
        elif question_idx in ["3","36","46"]:
            if choice== "A":
                Hedonism += 0 
            elif choice == "B":
                Hedonism += 1
            elif choice == "C":
                Hedonism += 2
            elif choice == "D":
                Hedonism += 3
            elif choice == "E":
                Hedonism += 4
            elif choice == "F":
                Hedonism += 5
        elif question_idx in ["17","32","48"]:
            if choice== "A":
                Achievement += 0 
            elif choice == "B":
                Achievement += 1
            elif choice == "C":
                Achievement += 2
            elif choice == "D":
                Achievement += 3
            elif choice == "E":
                Achievement += 4
            elif choice == "F":
                Achievement += 5
        elif question_idx in ["6","29","41","12","20","44"]:
            if choice== "A":
                Power += 0 
            elif choice == "B":
                Power += 1
            elif choice == "C":
                Power += 2
            elif choice == "D":
                Power += 3
            elif choice == "E":
                Power += 4
            elif choice == "F":
                Power += 5
        elif question_idx in ["13","26","53","2","35","50"]:
            if choice== "A":
                Security += 0 
            elif choice == "B":
                Security += 1
            elif choice == "C":
                Security += 2
            elif choice == "D":
                Security += 3
            elif choice == "E":
                Security += 4
            elif choice == "F":
                Security += 5
        elif question_idx in ["15","31","42","4","22","51"]:
            if choice== "A":
                Conformity += 0 
            elif choice == "B":
                Conformity += 1
            elif choice == "C":
                Conformity += 2
            elif choice == "D":
                Conformity += 3
            elif choice == "E":
                Conformity += 4
            elif choice == "F":
                Conformity += 5
        elif question_idx in ["18","33","40","7","38","54"]:
            if choice== "A":
                Tradition += 0 
            elif choice == "B":
                Tradition += 1
            elif choice == "C":
                Tradition += 2
            elif choice == "D":
                Tradition += 3
            elif choice == "E":
                Tradition += 4
            elif choice == "F":
                Tradition += 5
        elif question_idx in ["11","25","47","19","27","55"]:
            if choice== "A":
                Benevolence += 0 
            elif choice == "B":
                Benevolence += 1
            elif choice == "C":
                Benevolence += 2
            elif choice == "D":
                Benevolence += 3
            elif choice == "E":
                Benevolence += 4
            elif choice == "F":
                Benevolence += 5
        elif question_idx in ["8","21","45","5","37","52","14","34","57"]:
            if choice== "A":
                Universalism += 0 
            elif choice == "B":
                Universalism += 1
            elif choice == "C":
                Universalism += 2
            elif choice == "D":
                Universalism += 3
            elif choice == "E":
                Universalism += 4
            elif choice == "F":
                Universalism += 5

    return {"Self_Direction": Self_Direction,
            "Stimulation":Stimulation,
            "Hedonism":Hedonism,
            "Achievement":Achievement,
            "Power":Power,
            "Security":Security,
            "Conformity":Conformity,
            "Tradition":Tradition ,
            "Benevolence":Benevolence,
            "Universalism":Universalism}

def batch_run(model, prompt_data, personas, batch_size, dataset_name, prompt_options):
    for i, inst in enumerate(prompt_data):
        answer_list = []
        print("##############################",i,"/",len(prompt_data),"###########################################")
        for idx in range(0, len(personas), batch_size):
            temp_persons = personas[idx:idx + batch_size]
            prompt_list = []
            for per in temp_persons:
                prompt ="USER: "
                prompt += "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
                prompt += f"{per['description']} \n\n"
                prompt += "Use the given information to answer the question below. \n\n"
                prompt += inst['user_prompt']
                prompt_list.append(prompt)
            model_response = model.batch_respond(prompt_list)
            for response in model_response:
                answer = extract_answer(help_model, response, prompt_options)
                if answer!="NONE":
                    answer_list.append(answer)
            print("Respond:",model_response)
            print("Extract Answer:", answer)
        if len(answer_list)==0:
            answer_list.append("A")
        frequency = Counter(answer_list)    
        most_frequent_element = frequency.most_common(1)[0][0]
        inst['respond_answer'] = most_frequent_element
        inst['respond_list'] = answer_list
    
    return prompt_data


def single_run(model, prompt_data, personas, dataset_name, prompt_options):
    for i, inst in enumerate(prompt_data):
        print("##############################",i,"/",len(prompt_data),"###########################################")
        answer_list = []
        for p_idx, per in enumerate(personas):
            print("----------------------------------",p_idx,"/",len(personas),"-----------------------------------")

            prompt = "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
            prompt += f"{per['description']} \n\n"
            prompt += "Use the given information to answer the question below. \n\n"
            prompt += inst['user_prompt']
            prompt += "Which option would you choose?"
            messages = []
            messages.append({"role":"user", "content": prompt})
            model_response = model.respond(messages)
            answer = extract_answer(help_model, model_response, prompt_options)

            if answer!="NONE":
                answer_list.append(answer[0])
            print("Respond:",model_response)
            print("Extract Answer:", answer[0])
        if len(answer_list)==0:
            answer_list.append("A")
        frequency = Counter(answer_list)    
        most_frequent_element = frequency.most_common(1)[0][0]
        inst['respond_answer'] = most_frequent_element
        inst['respond_list'] = answer_list

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

    prompt_data, prompt_options  = init_dataset(args.dataset)


    if args.model in API_MODEL:
        model = LLM_API(args.model, api_key=args.config['openai_api_key'], base_url=args.config['base_url'])
    else:
        model = LLM_local(args.model, args.gpu)
    help_model =  LLM_API("gpt-3.5-turbo", api_key=args.config['openai_api_key'], base_url=args.config['base_url'])


    if args.role=="base":
        personas = generate_persona_description(args.role_num)
    elif args.role=="place":
        personas = generate_persona_occupation_description(args.role_num)

 
    prompt_data = single_run(model, prompt_data, personas, args.dataset, prompt_options)

    with open(args.output_dir+"/run_result.json", 'w') as file:
        json.dump(prompt_data, file, indent=4)

    if args.dataset=="mfq30":
        logs_info = """
        Evaluation Log
        --------------
        Evaluation Results:
        - Evaluation Date: {time}
        - Harm: {harm}
        - Fairness: {fairness}
        - Ingroup: {ingroup}
        - Authority: {authority}
        - Purity: {purity}
        """
        param_dict = evaluate_mfq30(prompt_data)
    elif args.dataset=="pvqrr":
        logs_info = """
        Evaluation Log
        --------------
        Evaluation Results:
        - Self-Direction: {Self_Direction}
        - Stimulation: {Stimulation}
        - Hedonism: {Hedonism}
        - Achievement: {Achievement}
        - Power: {Power}
        - Security: {Security}
        - Conformity: {Conformity}
        - Tradition: {Tradition}
        - Benevolence: {Benevolence}
        - Universalism: {Universalism}
        """
        param_dict = evaluation_pvqrr(prompt_data)

    logs_info = logs_info.format(**param_dict)
    # 写入文件
    with open(args.output_dir+'/logs.txt', 'a', encoding='utf-8') as file:
        file.write(logs_info)





