from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLM_Model:
    def __init__(self, args):
        
        self.gpu = args.gpu
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        #self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side="left"
        
        self.model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="float16")
        self.model.to(self.gpu)
        self.model.eval()

    def respond(self, user_prompt,  temperature=0.7, max_tokens=256, stop=None):
        
        # input_messages = [{"role": "user", "content": user_prompt}]
        # input_messages = self.tokenizer.apply_chat_template(
        #     input_messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        inputs = self.tokenizer(user_prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt")
        inputs = inputs.to(self.gpu)
    
        outputs = self.model.generate(**inputs,
                        max_new_tokens=max_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        num_return_sequences=1,
                        top_p=0.9,
                        temperature=temperature)
        results = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = results[len(user_prompt):]
        return answer
    
    def batch_respond(self, user_prompt, temperature=0.7, max_tokens=256, stop=None):

        inputs = self.tokenizer(user_prompt, padding=True, return_tensors="pt")
        inputs_on_gpu = {key: value.to(self.gpu) for key, value in inputs.items()}

        outputs = self.model.generate(**inputs_on_gpu,
                        max_new_tokens=max_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        num_return_sequences=1,
                        top_p=0.9,
                        temperature=temperature)
        
        results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results = [temp[len(user_prompt[idx]):] for idx,temp in enumerate(results)]

        return results
