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
API_MODEL = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]

def get_person_question(person_num, wenjuan_data):
    person_question_data = []
    for i, inst in enumerate(wenjuan_data):
        question = inst["question"]
        question_id = inst["question_id"]
        temp= {"question_id":question_id, "question":question, "person":None, "prompt":None}
        for p_idx, per in enumerate(personas):
            if p_idx>person_num:
                break
            prompt = "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
            prompt += f"{per['description']} \n\n"
            prompt += "Use the given information to answer the question below. \n\n"
            prompt += question

            temp['prompt'] = prompt
            temp['person'] = per['description']

            person_question_data.append(temp)

    return person_question_data


def init_dataset(datasets_name):
    
    if datasets_name=="mfq30":
        with open("Datasets/mfq-30/test2.json", 'r') as file:
            prompt_data = json.load(file)
    elif datasets_name=="pvqrr":
        with open("Datasets/pvq-rr/test2.json", 'r') as file:
            prompt_data = json.load(file)

    return prompt_data

def extract_answer(api_model, respond, options, origin_ques):

    extract_answer_example =  f"""
    Question: {origin_ques}
    Options: A. {options[0]} B. {options[1]} C. {options[2]} D. {options[3]} E. {options[4]} F. {options[5]}
    Model Response: {respond}
    Use the above information to determine which option the model's response is more consistent with.Please output the uppercase letters of the options directly, such as A,B,C,D,E,F and do not output anything else.
    """
    messages = []
    messages.append({"role":"user", "content": extract_answer_example})
    answer = api_model.respond(messages)
    extract_answer = answer.strip().upper()
    return extract_answer



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

    with open("Datasets/person_500.json", 'r') as file:
        person_500 = json.load(file)

    wenjuan_data = init_dataset(args.dataset)

    help_model =  LLM_API("gpt-3.5-turbo", api_key=args.config['openai_api_key'], base_url=args.config['base_url'])

    if args.role=="base":
        personas = generate_persona_description(args.role_num)
    elif args.role=="place":
        personas = generate_persona_occupation_description(args.role_num)

    person_question_data = get_person_question(person_num=50, wenjuan_data=wenjuan_data)




    with open(args.model_output, 'r') as file:
        prompt_data = json.load(file)

    if args.dataset=="mfq30":
        logs_info = """
        Evaluation Log
        --------------
        Evaluation Results:
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





