import random
import json


def BaseRun(args, instances, datasets, model, help_model):

    logs_result = []
    for idx, inst in enumerate(instances):
        print(idx,"/",len(instances))
        inst.model_response = model.respond(inst.user_prompt)
        inst.extract_answer = datasets.extract_answer(inst.model_response, help_model)
        logs_result.append(inst.to_dict())
        print("Questions \n", inst.question)
        print("Answer: \n", inst.model_response)

    with open(args.output_dir+"run_experiment.json", 'w') as file:
        json.dump(logs_result, file, indent=4)
    args.run_results = args.output_dir+"run_experiment.json"

    return instances











