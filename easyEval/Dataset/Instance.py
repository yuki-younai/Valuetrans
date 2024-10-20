


class Instance:
    def __init__(self) -> None:
        self.question =  None
        self.question_idx = None
        self.options = None
        self.answer = None
        self.role = "base"

        self.user_prompt = None
        self.model_response = None
        self.model_response_logprobs = None
        self.extract_answer = None
        self.response_correct = False
        self.response_evaluator = None
        self.response_fre = []
    def show(self):
        
        print("Question:", self.question)
        print("Options:", self.options)
        print("Answer:",self.answer)
        print("Prompt:",self.user_prompt)
        print("Response:",self.model_response)
    def to_dict(self):
        
        output = dict(question = self.question,
                      role = self.role ,
                      question_idx = self.question_idx,
                      options = self.options,
                      answer = self.answer,
                      user_prompt = self.user_prompt,
                      model_response = self.model_response,
                      extract_answer = self.extract_answer,
                      response_correct = self.response_correct,
                      response_fre = self.response_fre)
        return output
        