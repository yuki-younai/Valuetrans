




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

