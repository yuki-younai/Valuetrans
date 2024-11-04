import json
import copy
import argparse
import re
import json
from datetime import datetime
from collections import Counter
import sys
import time
from openai import OpenAI
import re
from src.role_person_generation import generate_persona_description
from src.role_occupation_generation import generate_persona_occupation_description
import pandas as pd
import json
import time

with open("Datasets/prompt/exp3_rewrite_respond.txt", 'r') as file:
    rewrite_prompt_format = file.read()

with open("Datasets/prompt/exp3_rewrire_evaluate.txt", 'r') as file:
    evaluate_prompt_format = file.read()

grained_decompos = {}
grained_decompos["Achievement"] = ["\n1. Competence:\n- Positive Indicator: Emphasizes the importance of demonstrating skill and proficiency in tasks (e.g., excelling in academic or professional settings).\n- Negative Indicator: Shows indifference towards skill development or a lack of effort in demonstrating competence.",
                                "\n2. Ambition:\n- Positive Indicator: Reflects a strong desire to achieve and excel (e.g., setting high goals and striving to reach them).\n- Negative Indicator: Indicates complacency or a lack of motivation to pursue personal success.",
                                "\n3. Success:\n- Positive Indicator: Values tangible accomplishments and recognition for achievements (e.g., receiving awards or promotions).\n- Negative Indicator: Downplays the significance of success or equates it with superficial achievements.",
                                "\n4. Social Approval:\n Positive Indicator: Acknowledges the importance of gaining recognition and respect from others (e.g., seeking validation from peers and society).\n - Negative Indicator: Dismisses the need for social recognition, prioritizing personal satisfaction over external validation.",
                                "\n5. Influence:- \nPositive Indicator: Supports the idea of using competence to have a positive impact on others and the community (e.g., leading teams or initiatives).\n- Negative\n Indicator: Suggests a lack of concern for making a difference or influencing others positively."]
grained_decompos["Benevolence"] = ["\n1.Helpfulness:\nPositive Indicator: Emphasizes willingness to assist others and contribute positively to their lives (e.g., offering support or aid to those in need). \nNegative Indicator: Displays indifference to others’ needs, showing reluctance or unwillingness to help when possible.",
                                   "\n2.Honesty:\nPositive Indicator: Values truthfulness and openness in interactions, fostering trust within the group (e.g., being straightforward in communication).\nNegative Indicator: Shows dishonesty or deceit, potentially harming relationships and reducing trust.",
                                   "\n3.Forgiveness:\nPositive Indicator: Encourages letting go of grudges and fostering reconciliation (e.g., forgiving mistakes to maintain harmony).\nNegative Indicator: Holds onto resentments or seeks retribution, which may lead to discord within the group.",
                                   "\n4.Responsibility:\nPositive Indicator: Demonstrates accountability for one’s actions, contributing to the well-being of the group (e.g., fulfilling promises and obligations).\nNegative Indicator: Neglects responsibilities, potentially disrupting group harmony or causing others to bear additional burdens.",
                                   "\n5.Loyalty:\nPositive Indicator: Shows commitment to supporting and standing by close relationships (e.g., being reliable in times of need).\nNegative Indicator: Displays disloyalty or unreliability, which can weaken bonds within the group."]
grained_decompos["Conformity"] = ["\n1.Obedience: \nPositive Indicator: Emphasizes compliance with rules, guidelines, and social norms (e.g., following instructions or respecting authority figures).\nNegative Indicator: Shows disregard for rules or authority, displaying defiance or challenging established norms.",
                                  "\n2.Self-Discipline:\nPositive Indicator: Values control over one’s actions and impulses to maintain order and respect (e.g., exercising restraint in difficult situations).\nNegative Indicator: Displays impulsiveness or lack of control, potentially causing disruptions or disregarding the expectations of others.",
                                  "\n3.Politeness:\nPositive Indicator: Reflects a courteous and respectful approach in social interactions (e.g., using respectful language and gestures).\nNegative Indicator: Shows rudeness or disrespect, failing to consider the feelings and comfort of others.",
                                  "\n4.Honoring Parents and Elders:\nPositive Indicator: Emphasizes respect and consideration for the guidance and values of parents and elders (e.g., consulting elders or valuing their wisdom).\nNegative Indicator: Ignores or dismisses the views of parents and elders, showing a lack of reverence for tradition or authority.",
                                  "\n5.Loyalty:\nPositive Indicator: Demonstrates dedication and commitment to close relationships and group obligations (e.g., standing by family, friends, or team members).\nNegative Indicator: Shows disloyalty or unreliability, potentially betraying the trust and expectations of close others."]

values_decrip = {}
values_decrip['Achievement'] = "Creativity is the value that drives the generation of novel and valuable ideas, solutions, or products through original thinking, embodying characteristics such as innovation, uniqueness, and societal or personal utility."
values_decrip['Benevolence'] = "Benevolence is a valuable and essential quality that can make a significant positive impact on individuals and society. "
values_decrip['Conformity'] = "Conformity refers to the act of adhering to or matching the behavior, beliefs, or attitudes of a group or society, often in order to avoid social pressure, exclusion, or punishment. It is a fundamental social phenomenon that plays a crucial role in maintaining social order and cohesion."


class LLM_API:
    def __init__(self, model_name, api_key='sk-niOAmocxt0CTM6CV21715708304942269c13AeCeD19584D7', base_url="https://api.claudeshop.top/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    def respond(self,  messages, temperature=0.7, max_tokens=128, stop=None):

        repeat_num = 0
        response_data = None
        while response_data == None:
            repeat_num += 1
            if repeat_num>5:
                response_data = "I Don't Know!"
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages= messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    stop=stop
                )
                response_data = completion.choices[0].message.content
                break
            except KeyboardInterrupt:
                sys.exit()
            except:
                print("Request timed out, retrying...")
                
        return response_data



def rewite_answer(values, agree_level, question, answer):

    grained_decrip = ''.join(grained_decompos[values])
    rewrite_prompt = rewrite_prompt_format.replace("{value}", values)
    rewrite_prompt = rewrite_prompt.replace("{description}", values_decrip[values])
    rewrite_prompt = rewrite_prompt.replace("{grained_description}", grained_decrip)
    rewrite_prompt = rewrite_prompt.replace("{agree_level}", agree_level)
    rewrite_prompt = rewrite_prompt.replace("{question}", question)
    rewrite_prompt = rewrite_prompt.replace("{answer}", answer)

    messages = []
    messages.append({"role": "user", "content": rewrite_prompt})
    new_answer = llm_model.respond(messages)

    match = re.search(answer_pattern, new_answer)
    if match:
        new_answer = match.group(0)

    return new_answer


def person_answer_evaluate(values, grained, person_prompt, new_answer):

    grained_person_prompt = person_prompt + evaluate_prompt_format.replace("{value}", values)
    grained_person_prompt = grained_person_prompt.replace("{description}", values_decrip[values])
    grained_person_prompt = grained_person_prompt.replace("{grained_description}", grained)
    grained_person_prompt = grained_person_prompt.replace("{statement}", new_answer)
    messages = []
    messages.append({"role": "user", "content": grained_person_prompt})
    evaluate_response = llm_model.respond(messages)

    if "agree" in evaluate_response.lower():
        evaluate_score = 1
    elif "neutral" in evaluate_response.lower():
        evaluate_score = 0
    elif "disagree" in evaluate_response.lower():
        evaluate_score = -1
    else:
        evaluate_score = 0

    return evaluate_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #1.基本选择：选择模型，数据集，模型配置，训练方式
    parser.add_argument("--dataset", type=str, default="Datasets/select_500_qa/ACHIEVEMENT_500QA.json",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--values", type=str, default="Achievement",
                        help="Path of the preprocessed dataset.")
    parser.add_argument("--gpu", type=int, default=3,
                    help="Random seed.")
    parser.add_argument("--role_num", type=int, default=10,
                    help="Random seed.")
    
    args = parser.parse_args()

    llm_model = LLM_API(model_name="/data2/szs/model_weights/Qwen2.5-32B-Instruct", api_key="token-abc123",base_url="http://localhost:8000/v1/")

    messages = [{"role": "system", "content": "You are a helpful Assistant"},
                {"role": "user", "content": "hello"}]
    print(llm_model.respond(messages))


    # 打开JSON文件并读取数据
    with open(args.dataset, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)

    max_rewrite_num = 3
    role_num = args.role_num
    values = args.values 
    output_name = values+"_rewrite_500QA.json"
    question_answer = []
    with open(output_name, 'w') as file:
        json.dump(question_answer, file, indent=4)


    agree_levels = ["Strongly disagree",  "Somewhat disagree", "Neutral" , "Somewhat agree", "Strongly agree"]
    agree_score_base = [-2, -1, 0 , 1, 2]
    answer_pattern = r'(?<="New_Answer": ")[^"]*"'
    evaluate_pattern = r'(?<="levels": ")[^"]*"'
    for idx, qa in enumerate(qa_data):
        print("#########################",idx,"/",len(qa_data),"###########################################################")
        question = qa['question']
        answer = qa['answer']
        template = {"prompt": question}
        time1 = time.time()
        for agree_idx, agree_level in enumerate(agree_levels):
            print("=======================================",agree_level,"==========================================")
            best_score = 999
            best_answer = None
            for j in range(max_rewrite_num):
                #1.重写回答
                print("--------------------------------New Rewrite Answer-------------------------------------------------------")
                print("Question:", question)
                print("Answer:", answer)
                new_answer = rewite_answer(values, agree_level, question, answer)
                print("New_Answer:", new_answer)
                print("---------------------------------Start person Evaluate-----------------------------------------------")
                #2.生成k个角色，分别对细粒度价值观进行评审
                personas = generate_persona_occupation_description(role_num)
                evaluate_scores = [0]*len(grained_decompos[values])
                for per in personas:
                    person_prompt ="USER: "
                    person_prompt += "Let's role-play. I will ask you a question and you must give me an answer. I want you to act as the person described below. Think from the person's perspective. \n"
                    person_prompt += f"{per['description']} \n"
                    person_prompt += "Use the given information to answer the question below. \n"
                    for grain_idx, grained in enumerate(grained_decompos[values]):
                        evaluate_score = person_answer_evaluate(values, grained, person_prompt, new_answer)
                        evaluate_scores[grain_idx] += evaluate_score 
                    print(evaluate_scores)
                #3.当分数差距小于0.3时，跳出
                agree_score = 2 * sum(evaluate_scores)/ (role_num) / len(evaluate_scores)
                if abs(agree_score-agree_score_base[agree_idx])<abs(best_score-agree_score_base[agree_idx]):
                    best_score = agree_score
                    best_answer = new_answer
                print("Finally_score:", agree_score)
                print("Best_score:", best_score)
                print("Best_Answer:", best_answer)
                
                if abs(best_score-agree_score_base[agree_idx])<0.5:
                    break
            print("Fianlly_Best_Answer For",agree_level,":", best_answer)   
            template["response_"+str(agree_idx+1)] = best_answer

        with open(output_name, 'r', encoding='utf-8') as file:
            question_answer = json.load(file)
        question_answer.append(template)
        with open(output_name, 'w') as file:
            json.dump(question_answer, file, indent=4)
        time2 = time.time()
        print("!!!!!!!!!!!!!!!!!Time Cost",time2-time1,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        break
















