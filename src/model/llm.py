import time
from openai import OpenAI
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

class LLM_local:
    def __init__(self, model_name_or_path, gpu):

        # Load Tokenizer
        print(">>> 1. Loading Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            add_eos_token= True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.chat_template is None:
            tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ 'USER: ' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ 'SYSTEM: ' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ 'ASSISTANT: '  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ 'ASSISTANT: ' }}\n{% endif %}\n{% endfor %}"
        print(">>> 2. Loading Model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype="bfloat16"
        )
        model.to(gpu)
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.gpu = gpu
                                
    def respond(self, messages, temperature=0.7, do_sample=True, max_tokens=1024, stop=None):

        generation_kwargs = {
                    "min_length": -1,
                    "temperature":temperature,
                    "top_k": 0.0,
                    "top_p": 1.0,
                    "do_sample": do_sample,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "max_new_tokens": max_tokens}
        input_messages = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(input_messages, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
        inputs = inputs.to(self.gpu)

        outputs = self.model.generate(**inputs, **generation_kwargs)
        results = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        model_response = results[len(input_messages):]
        
        return model_response