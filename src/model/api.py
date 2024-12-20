from transformers import HfArgumentParser, AutoModelForCausalLM
import time
from openai import OpenAI
import sys


class LLM_API:
    def __init__(self, model_name, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    def respond(self, messages, temperature=0.2, max_tokens=128, stop=None):

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
                    timeout=15,
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


