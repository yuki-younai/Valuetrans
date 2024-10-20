import random
import json
from collections import Counter


def VoteRun(args, instances, datasets, model, help_model, repeat_num = 5):

    logs_result = []
    for idx, inst in enumerate(instances):
        print(idx,"/",len(instances))
        response_list = []
        answer_list = []
        for i in range(repeat_num):
            model_response = model.respond(inst.user_prompt)
            extract_answer = datasets.extract_answer(inst.model_response, help_model)
            response_list.append(model_response)
            answer_list.append(extract_answer)

        frequency = Counter(answer_list)
        most_frequent_element = frequency.most_common(1)[0][0]
        indices = [index for index, element in enumerate(answer_list) if element == most_frequent_element]

        inst.model_response = response_list[indices[0]]
        inst.extract_answer = most_frequent_element
        
        logs_result.append(inst.to_dict())
        print("Questions \n", inst.question)
        print("Answer: \n", inst.model_response)

    with open(args.output_dir+"run_experiment.json", 'w') as file:
        json.dump(logs_result, file, indent=4)
    args.run_results = args.output_dir+"run_experiment.json"

    return instances











