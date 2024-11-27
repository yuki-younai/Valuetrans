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
grained_decompos["Hedonism"] = ["\n1.Pleasure:- \nPositive Indicator: Emphasizes the pursuit of enjoyment and gratification (e.g., seeking pleasurable experiences, such as good food or entertainment).- \nNegative Indicator: Suggests avoidance of pleasure or a focus on duty and obligation over enjoyment.",
                                "\n2.Enjoying Life:\n- Positive Indicator: Reflects an attitude of embracing life’s joys and making the most of experiences (e.g., celebrating achievements or indulging in leisure activities).\n- Negative Indicator: Indicates a serious or overly cautious approach to life, neglecting opportunities for enjoyment.",
                                "\n3.Self-Indulgence:\n- Positive Indicator: Supports the idea of treating oneself and indulging in desires (e.g., allowing oneself luxury or comfort).\n- Negative Indicator: Promotes self-denial or asceticism, discouraging the enjoyment of personal desires.",
                                "\n4.Sensuous Gratification:\nPositive Indicator: Values physical pleasure and sensory experiences (e.g., appreciating art, music, or nature).\n- Negative Indicator: Dismisses sensory experiences as unimportant or trivial, focusing instead on abstract or intellectual pursuits.",
                                "\n5.Immediate Satisfaction:\n- Positive Indicator: Encourages seeking immediate pleasure and fulfillment (e.g., acting on impulses that bring joy).\n- Negative Indicator: Advocates for long-term planning at the expense of immediate enjoyment, prioritizing future benefits over present pleasures."]

grained_decompos["Power"] = ["\n1.Authority:\nPositive Indicator: Emphasizes the importance of holding a position of control or decision-making (e.g., exercising influence over others in a leadership role).\nNegative Indicator: Dismisses the need for authority or expresses discomfort with leadership and decision-making roles.",
                             "\n2.Wealth:\nPositive Indicator: Reflects the pursuit of financial success and the accumulation of resources as symbols of status (e.g., seeking financial independence or luxury).\nNegative Indicator: Shows disinterest in wealth or financial achievement, potentially prioritizing non-material goals over financial success.",
                             "\n3.Social Power:\nPositive Indicator: Values the ability to influence others and affect outcomes within a community or organization (e.g., advocating for policies or changes that reflect one’s perspective).\nNegative Indicator: Avoids engaging in social influence or is indifferent to opportunities for impacting others’ views and actions.",
                             "\n4.Status and Prestige:\nPositive Indicator: Recognizes the importance of social recognition and being viewed with high regard by others (e.g., seeking positions that are respected or admired).\nNegative Indicator: Rejects the pursuit of social status, suggesting that external recognition is unimportant or irrelevant.",
                             "\n5.Dominance over People and Resources:\nPositive Indicator: Encourages the responsible control over people or resources to achieve social order or personal goals (e.g., managing a team or allocating resources strategically).\nNegative Indicator: Shows reluctance to control or manage others, indicating a preference for equal standing rather than hierarchy."]
grained_decompos["Security"] = ["\n1.Safety:\nPositive Indicator: Emphasizes the need for physical and emotional protection for oneself and others (e.g., promoting safety protocols or secure environments).\nNegative Indicator: Shows indifference to safety or advocates for taking unnecessary risks, potentially endangering self or others.",
                             "\n2.Harmony:\nPositive Indicator: Reflects a commitment to peaceful, cooperative relationships and minimizing conflict (e.g., encouraging teamwork and understanding).\nNegative Indicator: Displays a preference for conflict or dismisses the importance of harmonious relationships.",
                             "\n3.Stability:\nPositive Indicator: Values consistency and reliability in societal and personal contexts (e.g., advocating for stable job roles or predictable environments).\nNegative Indicator: Embraces instability or unpredictability, undermining consistency in personal or societal contexts.",
                             "\n4.Social Order:\nPositive Indicator: Supports structures and systems that maintain societal order and prevent chaos (e.g., respecting laws or community rules).\nNegative Indicator: Rejects or disregards social norms and rules, which may threaten social stability and order.",
                             "\n5.Family Security:\nPositive Indicator: Emphasizes the importance of protecting and caring for one’s family and close relationships (e.g., taking measures to ensure the well-being of family members).\nNegative Indicator: Shows neglect or lack of concern for the safety and well-being of family or loved ones."]
grained_decompos["Self_Direction"] = ["\n1.Autonomy:\n- Positive Indicator: Emphasizes individual choice and the ability to act independently (e.g., making personal decisions).\n- Negative Indicator: Emphasizes conformity or reliance on external authority for decision-making.",
                                      "\n2.Creativity:- Positive Indicator: Encourages innovation and original thought (e.g., valuing unique ideas).- Negative Indicator: Discourages original thought or insists on following established norms without questioning.",
                                      "\n3.Curiosity:- Positive Indicator: Reflects a desire to explore and learn (e.g., openness to new experiences).- Negative Indicator: Shows indifference to new ideas or experiences, suggesting a closed mindset.",
                                      "\n4 Goal Orientation:- Positive Indicator: Supports the idea of setting and pursuing personal goals (e.g., defining one's objectives).- Negative Indicator: Implies a lack of personal goals or encourages passivity in life choices.",
                                      "\n5.Mastery and Control:- Positive Indicator: Expresses a sense of control over actions and outcomes (e.g., proactive approach to challenges).- Negative Indicator: Suggests helplessness or dependence on others for direction and outcomes."]
grained_decompos["Stimulation"] =  ["\n1.Excitement:- Positive Indicator: Emphasizes the pursuit of thrilling experiences and emotional highs (e.g., seeking adventure or new activities).- Negative Indicator: Reflects a preference for routine or mundane experiences, avoiding anything that might provoke excitement.",
                                    "\n2.Novelty:- Positive Indicator: Encourages exploration of new ideas, places, and experiences (e.g., trying unfamiliar foods or traveling to new locations).- Negative Indicator: Shows resistance to change or a strong preference for the familiar and predictable.",
                                    "\n3.Challenge:- Positive Indicator: Supports taking risks and facing obstacles as a way to grow and learn (e.g., embracing difficult tasks or competitions).- Negative Indicator: Discourages taking risks or attempting difficult challenges, promoting comfort over growth.",
                                    "\n4.Variety:- Positive Indicator: Values a diverse range of experiences and activities to prevent boredom (e.g., engaging in multiple hobbies or interests).- Negative Indicator: Indicates a desire for uniformity and consistency, avoiding diverse experiences.",
                                    "\n5.Daring:- Positive Indicator: Encourages boldness and a willingness to step outside comfort zones (e.g., trying extreme sports or unconventional pursuits).- Negative Indicator: Promotes caution and a tendency to play it safe, avoiding situations that might be perceived as risky."]

values_decrip = {}
values_decrip['Achievement'] = "Creativity is the value that drives the generation of novel and valuable ideas, solutions, or products through original thinking, embodying characteristics such as innovation, uniqueness, and societal or personal utility."
values_decrip['Benevolence'] = "Benevolence is a valuable and essential quality that can make a significant positive impact on individuals and society. "
values_decrip['Conformity'] = "Conformity refers to the act of adhering to or matching the behavior, beliefs, or attitudes of a group or society, often in order to avoid social pressure, exclusion, or punishment. It is a fundamental social phenomenon that plays a crucial role in maintaining social order and cohesion."
values_decrip['Hedonism'] = "Hedonism is an adjective derived from the Greek word “hedone,” meaning pleasure. It refers to a school of thought that posits pleasure as the primary intrinsic good and the ultimate basis of morality. In other words, hedonism is the belief that the pursuit of pleasure and the avoidance of pain are the most important goals in life. "
values_decrip['Power'] = "Power is the multifaceted ability to control, influence, or direct outcomes, existing in various forms such as political, economic, and social, and is relative and dynamic, often formalized as authority; it can be potential or actual, and while essential for achieving objectives, its legitimacy and responsible use are critical to prevent abuse."
values_decrip['Security'] = "Security refers to the measures and systems put in place to protect against, detect, and respond to threats, ensuring the safety, integrity, and confidentiality of assets, whether physical, digital, or personal, thereby providing a state of stability and peace of mind for individuals, organizations, and nations."
values_decrip['Self_Direction'] = "“Self-Direction” refers to the ability and practice of guiding oneself without relying heavily on external input or direction. It involves a combination of personal initiative, autonomy, and self-management."
values_decrip["Stimulation"] = "Stimulation refers to the action or process of arousing interest, excitement, or activity in someone or something. It is the act of stimulating, which can be applied to various contexts and can affect the body, mind, or senses. "
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

    llm_model = LLM_API(model_name="/data2/szs/model_weights/Qwen2.5-32B-Instruct", api_key="token-abc123",base_url="http://localhost:8220/v1/")

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
    #question_answer = []
    # with open(output_name, 'w') as file:
    #     json.dump(question_answer, file, indent=4)


    agree_levels = ["Strongly disagree",  "Somewhat disagree", "Neutral" , "Somewhat agree", "Strongly agree"]
    agree_score_base = [-2, -1, 0 , 1, 2]
    answer_pattern = r'(?<="New_Answer": ")[^"]*"'
    evaluate_pattern = r'(?<="levels": ")[^"]*"'
    for idx, qa in enumerate(qa_data):
        if idx<300:
            continue
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


















