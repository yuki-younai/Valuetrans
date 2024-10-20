import json
import copy
import argparse
from easyEval.utils.utils import init_output, set_seed
from easyEval.Dataset import str2datasets
from easyEval.Model import API_MODEL, str2models
from easyEval.Run import str2runs
from easyEval.Evaluation import str2evaluation
from easyEval.utils.role_person_generation import generate_persona_description
from easyEval.utils.role_occupation_generation import generate_persona_occupation_description
import re
import json
from datetime import datetime
from collections import Counter


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #1.基本选择：选择模型，数据集，模型配置，训练方式
    parser.add_argument("--dataset", type=str, default="mfq30",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--method", type=str, default="base",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--role", type=str, default="place",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--run_results", type=str, default=None,
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--ratio", type=float, default=0.1,
                    help="Random seed.")
    parser.add_argument("--gpu", type=int, default=3,
                    help="Random seed.")
    parser.add_argument("--role_num", type=int, default=2,
                    help="Random seed.")
    batch_size = 10
    args = parser.parse_args()
    with open("easyEval/config.json", 'r') as file:
        args.config = json.load(file)

    ################
    #Init Environment
    ################
    set_seed(42)
    init_output(args)
    ################
    #Init Dataset And Model
    ################
    dataset = str2datasets[args.dataset]()
    if args.model in API_MODEL:
        model = str2models["api"](args.model, api_key=args.config['openai_api_key'], base_url=args.config['base_url'])
    else:
        model = str2models["local"](args)
    help_model = str2models["api"]("gpt-3.5-turbo", api_key=args.config['openai_api_key'], base_url=args.config['base_url'])
    batch_eval = hasattr(model, "batch_respond")
    ################
    #Run Experiment
    ################  
    instances = dataset.init_instance(args.ratio)

    if args.role=="base":
        personas = generate_persona_description(args.role_num)
    elif args.role=="place":
        personas = generate_persona_occupation_description(args.role_num)

    logs_result = []
    for i, inst in enumerate(instances):
        respond_list = []
        answer_list = []
        if batch_eval:
            for idx in range(0, len(personas), batch_size):
                print(i,"/",len(instances), " ", idx,"/",len(personas))
                temp_persons = personas[idx:idx + batch_size]
                prompt_list = []
                for per in temp_persons:
                    prompt ="USER: "
                    prompt += "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
                    prompt += f"{per['description']} \n\n"
                    prompt += "Use the given information to answer the question below. \n\n"
                    prompt += inst.user_prompt
                    prompt += "ASSISTANT:"
                    prompt_list.append(prompt)
                model_response = model.batch_respond(prompt_list)
                for response in model_response:
                    extract_answer = dataset.extract_answer(response, help_model)
                    if extract_answer!="NONE":
                        answer_list.append(extract_answer)
                        respond_list.append(response)
        else:
            for idx, persona in enumerate(personas):
                prompt ="USER: "
                prompt += "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n\n"
                prompt += f"{persona['description']} \n\n"
                prompt += "Use the given information to answer the question below. \n\n"
                prompt += inst.user_prompt
                prompt += "ASSISTANT:"
                print(i,"/",len(instances), " ", idx,"/",len(personas))
                model_response = model.respond(prompt)
                extract_answer = dataset.extract_answer(model_response, help_model)
                if extract_answer!="NONE":
                    answer_list.append(extract_answer)
                    respond_list.append(model_response)

        if len(answer_list)==0:
            answer_list.append("A")
        frequency = Counter(answer_list)    
        most_frequent_element = frequency.most_common(1)[0][0]
        #print("Response",respond_list)
        print("Answer_List", answer_list)
        print("Answer Freqency",most_frequent_element)
        #indices = [index for index, element in enumerate(answer_list) if element == most_frequent_element]
        inst.user_prompt = prompt
        inst.extract_answer = most_frequent_element
        inst.response_fre = answer_list

        logs_result.append(inst.to_dict())

    with open(args.output_dir+"0_run_experiment.json", 'w') as file:
        json.dump(logs_result, file, indent=4)

    ################
    #Evaluation
    ################  
    evaluation = str2evaluation[dataset.type](args, dataset, model, help_model )
    args.run_results = args.output_dir+"0_run_experiment.json"
    evaluation.run(args.run_results)


if __name__ == "__main__":
    main()




















